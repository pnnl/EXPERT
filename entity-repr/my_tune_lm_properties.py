#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=masked-lm
"""
# You can also adapt this script on your own mlm task. Pointers for this are left as comments.

import argparse
import logging
import math
import os
import random
from pathlib import Path

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator, DistributedType
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    SchedulerType,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import get_full_repo_name
from transformers.utils.versions import require_version

from my_utils import *
import numpy as np
import pickle
import json
import copy
from collections import defaultdict
from torch.nn import CrossEntropyLoss


logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Masked Language Modeling task")
    # Han: many arguments below will not be used, but keeping for future edits
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the test data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--raw_data_percentage",
        default=100,
        help="The percentage of raw data used as the train set",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--train_data_cap",
        type=int,
        default=2,
        help="Max number of data for which we will save graidents.",
    )
    parser.add_argument(
        "--eval_data_cap",
        type=int,
        default=2,
        help="Max number of data for which we will save graidents.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.",
    )
    parser.add_argument(
        "--line_by_line",
        type=bool,
        default=False,
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument("--no_save_grads", action="store_true", help="Whether to save gradients to a file.")
    # for computing influence scores w.r.t. the querying file
    parser.add_argument(
        "--query_file", type=str, default=None, help="A pickle file containing gradient information from the querying data."
    )
    parser.add_argument(
        "--query_data_cap", type=int, default=None, help="Max number of data for which we will save gradients.",
    )
    parser.add_argument("--influence_metric", type=str, default=None, help="Metric for computing the gradients.")
    parser.add_argument(
        "--tokenized_data_file_path", type=str, default=None, help="Path of the tokenized data file."
    )
    parser.add_argument(
        "--if_create_tokenized_data_file", type=str, default=None, help="Whether to create a new tokenized data file (yes or no)."
    )
    parser.add_argument(
        "--context_sentence_k", type=int, default=0, help="k sentences before and after the entity as context.",
    )
    parser.add_argument(
        "--data_pairs_k", type=int, default=5, help="k pairs of examples for each example in the data",
    )
    parser.add_argument(
        "--align_weight_over_uniform", type=float, default=0.5, help="balancing loss factor",
    )
    parser.add_argument(
        "--repr_mode", type=str, default=None, help="embedding or gradient representation"
    )
    parser.add_argument(
        "--log_epoch_interval", type=int, default=1, help="save model after how many epochs",
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    args.logger = logger

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    assert args.use_slow_tokenizer == True # Han: for a compatible tokenizer with the prompt tuning model
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # Loading model
    if args.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)
    # # Loading model
    # if args.tuned_model_dir:
    #     if os.path.exists(os.path.join(args.output_dir, 'pytorch_model.bin')):
    #         previous_state_dict = torch.load(os.path.join(args.output_dir, 'pytorch_model.bin'))
    #     else:
    #         previous_state_dict = OrderedDict()
    #     distant_state_dict = torch.load(os.path.join(args.tuned_model_dir, 'pytorch_model.bin'))
    #     previous_state_dict.update(distant_state_dict) # final layer of previous model and distant model need different attribute names
    #     model = AutoModelForSequenceClassification.from_pretrained(
    #         args.tuned_model_dir,
    #         state_dict=previous_state_dict,
    #         config=config,
    #     )

    model.resize_token_embeddings(len(tokenizer))

    # Tokenize data (cached)
    jump = False
    if args.tokenized_data_file_path and args.if_create_tokenized_data_file:
        if args.if_create_tokenized_data_file == "no":
            tokenized_datasets = pickle.load(open(args.tokenized_data_file_path, 'rb'))
            jump = True
        elif args.if_create_tokenized_data_file == "yes":
            if accelerator.is_main_process:
                pass
        else:
            raise ValueError("check args.if_create_tokenized_data_file")
    if not jump:
        tokenized_datasets = dict()
        tokenized_datasets['train'] = process_raw_dataset(args.train_file, tokenizer, args)
        tokenized_datasets['validation'] = process_raw_dataset(args.validation_file, tokenizer, args)
        tokenized_datasets['test'] = process_raw_dataset(args.test_file, tokenizer, args)
        if args.tokenized_data_file_path and args.if_create_tokenized_data_file:
            if args.if_create_tokenized_data_file == "yes":
                if accelerator.is_main_process:
                    pickle.dump(tokenized_datasets, open(args.tokenized_data_file_path, 'wb'))

    # Entity type to index
    enttype2idx = {enttype : _i for _i, enttype in enumerate(sorted(set([data_dict['entity_label'] for data_dict in tokenized_datasets['train']])))}
    idx2enttype = {_v : _k for _k, _v in enttype2idx.items()}
    logger.info(f"entity type dictionary {enttype2idx}")
    args.enttype2idx = enttype2idx
    args.idx2enttype = idx2enttype

    train_dataloader, train_align_dataloader, train_uniform_dataloader = \
        prepare_dataloaders(tokenized_datasets['train'], args.per_device_train_batch_size, args)
    eval_dataloader, eval_align_dataloader, eval_uniform_dataloader = \
        prepare_dataloaders(tokenized_datasets['validation'], args.per_device_eval_batch_size, args)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, \
        train_align_dataloader, train_uniform_dataloader, eval_align_dataloader, eval_uniform_dataloader = \
        accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader, \
        train_align_dataloader, train_uniform_dataloader, eval_align_dataloader, eval_uniform_dataloader)

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    ####
    # Evaluate properties
    model_copy = copy.deepcopy(model)
    evaluate_properties(model_copy, args, accelerator, eval_align_dataloader, eval_uniform_dataloader)
    ####

    # Scheduler and math around the number of training steps.
    args.actual_train_step_len = min(len(train_align_dataloader), len(train_uniform_dataloader), args.train_data_cap) # subject to change
    num_update_steps_per_epoch = math.ceil(args.actual_train_step_len / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {args.actual_train_step_len}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    if args.repr_mode == "embedding":
        # embedding alignment and uniformity training
        null_label_id = torch.LongTensor([-100]) # MLM non-label id is -100, can specify multiple
        null_label_id = null_label_id.to(accelerator.device).view(-1, 1) # utilize the broadcasting feature
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0
        for epoch in range(args.num_train_epochs):
            model.train()
            model.zero_grad()
            for step, (batch_align, batch_uniform) in enumerate(zip(train_align_dataloader, train_uniform_dataloader)):
                if step >= args.train_data_cap:
                    break
                align_input_ids_without_masks, align_input_ids_with_masks, align_attention_ids, align_recon_labels, align_entity_type = batch_align[:5]
                align_other_input_ids_without_masks, align_other_input_ids_with_masks, align_other_attention_ids, align_other_recon_labels, align_other_entity_type = batch_align[5:]
                uniform_input_ids_without_masks, uniform_input_ids_with_masks, uniform_attention_ids, uniform_recon_labels, uniform_entity_type = batch_uniform[:5]
                uniform_other_input_ids_without_masks, uniform_other_input_ids_with_masks, uniform_other_attention_ids, uniform_other_recon_labels, uniform_other_entity_type = batch_uniform[5:]

                align_batch1 = {'input_ids': align_input_ids_without_masks, \
                    'attention_mask': align_attention_ids, \
                    'labels': align_recon_labels, \
                    }
                align_batch2 = {'input_ids': align_other_input_ids_without_masks, \
                    'attention_mask': align_other_attention_ids, \
                    'labels': align_other_recon_labels, \
                    }
                uniform_batch1 = {'input_ids': uniform_input_ids_without_masks, \
                    'attention_mask': uniform_attention_ids, \
                    'labels': uniform_recon_labels, \
                    }
                uniform_batch2 = {'input_ids': uniform_other_input_ids_without_masks, \
                    'attention_mask': uniform_other_attention_ids, \
                    'labels': uniform_other_recon_labels, \
                    }
                assert len(align_batch1['input_ids']) == 1
                assert len(uniform_batch1['input_ids']) == 1
                align_idx_labels_1 = torch.where(align_batch1['labels'].view(1, -1).ne(null_label_id).sum(0))[0]
                align_idx_labels_2 = torch.where(align_batch2['labels'].view(1, -1).ne(null_label_id).sum(0))[0]
                uniform_idx_labels_1 = torch.where(uniform_batch1['labels'].view(1, -1).ne(null_label_id).sum(0))[0]
                uniform_idx_labels_2 = torch.where(uniform_batch2['labels'].view(1, -1).ne(null_label_id).sum(0))[0]

                # Han: use the last hidden states for interpretation
                outputs1 = torch.squeeze(accelerator.unwrap_model(model).bert(input_ids=align_batch1['input_ids'],attention_mask=align_batch1['attention_mask'])[0], 0)
                outputs2 = torch.squeeze(accelerator.unwrap_model(model).bert(input_ids=align_batch2['input_ids'],attention_mask=align_batch2['attention_mask'])[0], 0)
                repr_1, rcount_1 = 0, 0
                repr_2, rcount_2 = 0, 0
                for idx_of_loss in align_idx_labels_1:
                    repr_1 = repr_1 + outputs1[idx_of_loss]
                    rcount_1 += 1
                for idx_of_loss in align_idx_labels_2:
                    repr_2 = repr_2 + outputs2[idx_of_loss]
                    rcount_2 += 1
                repr_1 = repr_1 / rcount_1 # mean pooling
                repr_2 = repr_2 / rcount_2
                align_loss = align_property(repr_1, repr_2)

                outputs1 = torch.squeeze(accelerator.unwrap_model(model).bert(input_ids=uniform_batch1['input_ids'],attention_mask=uniform_batch1['attention_mask'])[0], 0)
                outputs2 = torch.squeeze(accelerator.unwrap_model(model).bert(input_ids=uniform_batch2['input_ids'],attention_mask=uniform_batch2['attention_mask'])[0], 0)
                repr_1, rcount_1 = 0, 0
                repr_2, rcount_2 = 0, 0
                for idx_of_loss in uniform_idx_labels_1:
                    repr_1 = repr_1 + outputs1[idx_of_loss]
                    rcount_1 += 1
                for idx_of_loss in uniform_idx_labels_2:
                    repr_2 = repr_2 + outputs2[idx_of_loss]
                    rcount_2 += 1
                repr_1 = repr_1 / rcount_1 # mean pooling
                repr_2 = repr_2 / rcount_2
                uniform_loss = uniform_property_no_exp(repr_1, repr_2) # not using exp when tuning

                loss = args.align_weight_over_uniform * align_loss + (1-args.align_weight_over_uniform) * uniform_loss
                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                if (step + 1) % args.gradient_accumulation_steps == 0 or step == args.actual_train_step_len - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1
                if completed_steps >= args.max_train_steps:
                    break
            model_copy = copy.deepcopy(model)
            evaluate_properties(model_copy, args, accelerator, eval_align_dataloader, eval_uniform_dataloader)
            if (epoch + 1) % args.log_epoch_interval == 0:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                epoch_dir = os.path.join(args.output_dir, f'epoch_{epoch + 1}')
                unwrapped_model.save_pretrained(epoch_dir, save_function=accelerator.save)
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(epoch_dir)
        accelerator.wait_for_everyone()
    elif args.repr_mode == "gradient":
        # gradient alignment and uniformity training
        loss_fct = CrossEntropyLoss()
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0
        for epoch in range(args.num_train_epochs):
            model.train()
            model.zero_grad()
            for step, (batch_align, batch_uniform) in enumerate(zip(train_align_dataloader, train_uniform_dataloader)):
                if step >= args.train_data_cap:
                    break
                align_input_ids_without_masks, align_input_ids_with_masks, align_attention_ids, align_recon_labels, align_entity_type = batch_align[:5]
                align_other_input_ids_without_masks, align_other_input_ids_with_masks, align_other_attention_ids, align_other_recon_labels, align_other_entity_type = batch_align[5:]
                uniform_input_ids_without_masks, uniform_input_ids_with_masks, uniform_attention_ids, uniform_recon_labels, uniform_entity_type = batch_uniform[:5]
                uniform_other_input_ids_without_masks, uniform_other_input_ids_with_masks, uniform_other_attention_ids, uniform_other_recon_labels, uniform_other_entity_type = batch_uniform[5:]

                align_batch1 = {'input_ids': align_input_ids_with_masks, \
                    'attention_mask': align_attention_ids, \
                    'labels': align_recon_labels, \
                    }
                align_batch2 = {'input_ids': align_other_input_ids_with_masks, \
                    'attention_mask': align_other_attention_ids, \
                    'labels': align_other_recon_labels, \
                    }
                uniform_batch1 = {'input_ids': uniform_input_ids_with_masks, \
                    'attention_mask': uniform_attention_ids, \
                    'labels': uniform_recon_labels, \
                    }
                uniform_batch2 = {'input_ids': uniform_other_input_ids_with_masks, \
                    'attention_mask': uniform_other_attention_ids, \
                    'labels': uniform_other_recon_labels, \
                    }
                assert len(align_batch1['input_ids']) == 1
                assert len(uniform_batch1['input_ids']) == 1

                # Han: calculate gradients
                outputs1 = accelerator.unwrap_model(model).bert(input_ids=align_batch1['input_ids'],attention_mask=align_batch1['attention_mask'])[0]
                scores1 = accelerator.unwrap_model(model).cls(outputs1)
                grads1 = torch.autograd.grad(loss_fct(scores1.view(-1, accelerator.unwrap_model(model).config.vocab_size), align_batch1['labels'].view(-1)), model.parameters(), create_graph=True, retain_graph=True)
                repr_1 = tensor_gather_flat_grads(grads1)
                outputs2 = accelerator.unwrap_model(model).bert(input_ids=align_batch2['input_ids'],attention_mask=align_batch2['attention_mask'])[0]
                scores2 = accelerator.unwrap_model(model).cls(outputs2)
                grads2 = torch.autograd.grad(loss_fct(scores2.view(-1, accelerator.unwrap_model(model).config.vocab_size), align_batch2['labels'].view(-1)), model.parameters(), create_graph=True, retain_graph=True)
                repr_2 = tensor_gather_flat_grads(grads2)
                align_loss = align_property(repr_1, repr_2)

                outputs1 = accelerator.unwrap_model(model).bert(input_ids=uniform_batch1['input_ids'],attention_mask=uniform_batch1['attention_mask'])[0]
                scores1 = accelerator.unwrap_model(model).cls(outputs1)
                grads1 = torch.autograd.grad(loss_fct(scores1.view(-1, accelerator.unwrap_model(model).config.vocab_size), uniform_batch1['labels'].view(-1)), model.parameters(), create_graph=True, retain_graph=True)
                repr_1 = tensor_gather_flat_grads(grads1)
                outputs2 = accelerator.unwrap_model(model).bert(input_ids=uniform_batch2['input_ids'],attention_mask=uniform_batch2['attention_mask'])[0]
                scores2 = accelerator.unwrap_model(model).cls(outputs2)
                grads2 = torch.autograd.grad(loss_fct(scores2.view(-1, accelerator.unwrap_model(model).config.vocab_size), uniform_batch2['labels'].view(-1)), model.parameters(), create_graph=True, retain_graph=True)
                repr_2 = tensor_gather_flat_grads(grads2)
                uniform_loss = uniform_property_no_exp(repr_1, repr_2) # not using exp when tuning

                loss = args.align_weight_over_uniform * align_loss + (1-args.align_weight_over_uniform) * uniform_loss
                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                if (step + 1) % args.gradient_accumulation_steps == 0 or step == args.actual_train_step_len - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1
                if completed_steps >= args.max_train_steps:
                    break
            model_copy = copy.deepcopy(model)
            evaluate_properties(model_copy, args, accelerator, eval_align_dataloader, eval_uniform_dataloader)
            if (epoch + 1) % args.log_epoch_interval == 0:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                epoch_dir = os.path.join(args.output_dir, f'epoch_{epoch + 1}')
                unwrapped_model.save_pretrained(epoch_dir, save_function=accelerator.save)
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(epoch_dir)
        accelerator.wait_for_everyone()
    else:
        raise ValueError("check repr_mode")

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
