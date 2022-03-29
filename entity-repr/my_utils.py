from collections import defaultdict
import numpy as np
import torch
from tqdm.auto import tqdm
import time
from contextlib import contextmanager

import math
import os
import random
from pathlib import Path

import datasets
from datasets import load_dataset
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator, DistributedType

import pickle
import json
import copy
from collections import defaultdict
from torch.nn import CrossEntropyLoss
from collections import OrderedDict
from datasets import load_metric

def align_property(repr_1, repr_2):
    assert len(repr_1.shape) == 1
    assert len(repr_2.shape) == 1
    _norm1 = torch.linalg.norm(repr_1, dim=0)
    _norm2 = torch.linalg.norm(repr_2, dim=0)
    repr_a = repr_1 / _norm1
    repr_b = repr_2 / _norm2
    diff = repr_a - repr_b
    norm = torch.linalg.norm(diff, dim=0)
    return norm ** 2

def uniform_property(repr_1, repr_2):
    assert len(repr_1.shape) == 1
    assert len(repr_2.shape) == 1
    _norm1 = torch.linalg.norm(repr_1, dim=0)
    _norm2 = torch.linalg.norm(repr_2, dim=0)
    repr_a = repr_1 / _norm1
    repr_b = repr_2 / _norm2
    diff = repr_a - repr_b
    norm = torch.linalg.norm(diff, dim=0)
    return torch.exp(-2 * (norm ** 2))

def uniform_property_no_exp(repr_1, repr_2):
    assert len(repr_1.shape) == 1
    assert len(repr_2.shape) == 1
    _norm1 = torch.linalg.norm(repr_1, dim=0)
    _norm2 = torch.linalg.norm(repr_2, dim=0)
    repr_a = repr_1 / _norm1
    repr_b = repr_2 / _norm2
    diff = repr_a - repr_b
    norm = torch.linalg.norm(diff, dim=0)
    return -1 * (norm ** 2)

def get_grads(loss, model, retain_graph=False):
    grads = torch.autograd.grad(loss, model.parameters(), retain_graph=retain_graph)
    return grads

def get_named_grads(grads, model):
    named_grads_dict = dict()
    for (n, p), g in zip(model.named_parameters(), grads):
        named_grads_dict[n] = g
    return named_grads_dict

def gather_flat_grads(grads):
    views = []
    for p in grads:
        if p.is_sparse:
            view = p.detach().to_dense().view(-1)
        else:
            view = p.detach().view(-1)
        views.append(view)
    cat = torch.cat(views, 0)
    return cat

def tensor_gather_flat_grads(grads):
    views = []
    for p in grads:
        if p.is_sparse:
            view = p.to_dense().view(-1)
        else:
            view = p.view(-1)
        views.append(view)
    cat = torch.cat(views, 0)
    return cat

def debug_grads(model, dataloader, tokenizer):
    for step, batch in enumerate(dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        test_grads = get_grads(loss, model)
        test_named_grads = get_named_grads(test_grads, model)
        test_flat_grads = gather_flat_grads(test_grads)
        breakpoint()

@contextmanager
def timing(description: str) -> None:
    start_time = time.time()
    yield
    print(f"#### Time for {description}: {time.time() - start_time} sec ####")


def convert_raw_to_features(word_list, entity_loc, max_seq_length, tokenizer, args):
    data_dict = dict()

    # tokenize each word and concatenate them
    bert_tokens = []
    entity_token_start_loc, entity_token_loc_length = None, None
    bert_tokens.append("[CLS]")
    for word_idx, word in enumerate(word_list):
        new_tokens = tokenizer.tokenize(word)
        if len(bert_tokens) + len(new_tokens) > max_seq_length - 1:
            if entity_token_start_loc is None:
                args.logger.info(f"skipping one entity since it's out of max_seq_length")
                return None # entity is out of max_seq_length, need to adjust context_sentence_k to include it
            break
        else:
            if word_idx >= entity_loc[0] and word_idx < entity_loc[1]:
                if entity_token_start_loc is None:
                    entity_token_start_loc = len(bert_tokens)
                    entity_token_loc_length = len(new_tokens)
                else:
                    entity_token_loc_length += len(new_tokens)
            bert_tokens.extend(new_tokens)
    bert_tokens.append("[SEP]")

    # mask entity span if needed
    bert_tokens_copy = copy.deepcopy(bert_tokens) # copy is without mask, original has masks
    for _i in range(entity_token_start_loc, entity_token_start_loc + entity_token_loc_length):
        bert_tokens[_i] = "[MASK]"

    input_ids = tokenizer.convert_tokens_to_ids(bert_tokens)
    copy_input_ids = tokenizer.convert_tokens_to_ids(bert_tokens_copy)
    input_mask = [1] * len(input_ids)
    padding_ids = [tokenizer.pad_token_id] * (max_seq_length - len(input_ids))
    padding_mask = [0] * (max_seq_length - len(input_ids))
    input_ids += padding_ids
    input_mask += padding_mask
    copy_input_ids += padding_ids

    label_ids = [-100] * max_seq_length
    for _i in range(entity_token_start_loc, entity_token_start_loc + entity_token_loc_length):
        label_ids[_i] = copy_input_ids[_i]

    assert len(input_ids) == max_seq_length
    assert len(copy_input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(label_ids) == max_seq_length

    data_dict['input_ids_with_masks'] = input_ids
    data_dict['input_ids_without_masks'] = copy_input_ids
    data_dict['attention_ids'] = input_mask
    data_dict['labels'] = label_ids
    return data_dict


def process_raw_dataset(dataset_path, tokenizer, args):
    # Han: start processing dataset
    # assume finished running `tar -xvzf scirex_dataset/release_data.tar.gz --directory scirex_dataset` in `./SciREX`
    # dict_keys(['coref', 'coref_non_salient', 'doc_id', 'method_subrelations', 'n_ary_relations', 'ner', 'sections', 'sentences', 'words'])
    train_data = [json.loads(jline) for jline in open(dataset_path, 'r').readlines()]

    processed_train_data = []
    for doc_i, raw_data in enumerate(train_data):
        if doc_i % 10 == 0:
            args.logger.info(f"processing the {doc_i}th document")
        ner_locs = raw_data['ner']
        sentence_locs = raw_data['sentences']
        idx_sent_searched = 0
        for ner_i, ner_loc in enumerate(ner_locs):
            data_dict = dict()
            target_sentence_loc = None
            for idx_sent in range(idx_sent_searched, len(sentence_locs)):
                sentence_loc = sentence_locs[idx_sent]
                if ner_loc[0] >= sentence_loc[0] and ner_loc[0] < sentence_loc[1]: # only using entity start loc
                    target_sentence_loc = idx_sent
                    # idx_sent_searched = idx_sent # if ner_locs and sentence_locs are sorted we can use this to improve efficiency
                    break
            assert target_sentence_loc is not None, breakpoint()

            context_start_loc = sentence_locs[max(0, target_sentence_loc - args.context_sentence_k)][0]
            context_end_loc = sentence_locs[min(len(sentence_locs) - 1, target_sentence_loc + args.context_sentence_k)][1]
            context_words = [raw_data['words'][_i] for _i in range(context_start_loc, context_end_loc)]

            data_dict['document_index'] = doc_i
            data_dict['entity_index'] = ner_i
            data_dict['entity_label'] = ner_loc[2]
            data_dict['entity_self_start_loc'] = ner_loc[0]
            data_dict['entity_self_end_loc'] = ner_loc[1]
            data_dict['entity_sentence_start_loc'] = sentence_locs[target_sentence_loc][0]
            data_dict['entity_sentence_end_loc'] = sentence_locs[target_sentence_loc][1]
            data_dict['entity_context_start_loc'] = context_start_loc
            data_dict['entity_context_end_loc'] = context_end_loc
            data_dict['context_words'] = context_words
            data_dict['entity_word_start_idx_in_context'] = ner_loc[0] - context_start_loc
            data_dict['entity_word_end_idx_in_context'] = ner_loc[1] - context_start_loc
            data_dict['entity_words'] = [context_words[_i] for _i in range(data_dict['entity_word_start_idx_in_context'], data_dict['entity_word_end_idx_in_context'])]

            feature_dict = convert_raw_to_features(data_dict['context_words'], (data_dict['entity_word_start_idx_in_context'], data_dict['entity_word_end_idx_in_context']), args.max_seq_length, tokenizer, args)
            if feature_dict is None: # aborting for one example
                continue
            data_dict['feature_input_ids_with_masks'] = feature_dict['input_ids_with_masks']
            data_dict['feature_input_ids_without_masks'] = feature_dict['input_ids_without_masks']
            data_dict['feature_attention_ids'] = feature_dict['attention_ids']
            data_dict['feature_labels'] = feature_dict['labels']
            processed_train_data.append(data_dict)
    return processed_train_data


def prepare_dataloaders(tokenized_dataset_split, batch_size, args): # tokenized_dataset_split can be tokenized_datasets['train']
    # Preparing data
    enttype2dataidx = defaultdict(list)
    input_ids_without_masks, input_ids_with_masks, attention_ids, recon_labels, entity_type = [], [], [], [], [] # 1,2,3,4,5
    for _i, data_dict in enumerate(tokenized_dataset_split):
        input_ids_without_masks.append(data_dict['feature_input_ids_without_masks'])
        input_ids_with_masks.append(data_dict['feature_input_ids_with_masks'])
        attention_ids.append(data_dict['feature_attention_ids'])
        recon_labels.append(data_dict['feature_labels'])
        entity_type.append(args.enttype2idx[data_dict['entity_label']])
        enttype2dataidx[args.enttype2idx[data_dict['entity_label']]].append(_i)

    train_dataset = TensorDataset(torch.LongTensor(input_ids_without_masks), \
        torch.LongTensor(input_ids_with_masks), \
        torch.LongTensor(attention_ids), \
        torch.LongTensor(recon_labels), \
        torch.LongTensor(entity_type))
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

    # alignment and uniformity data pairs
    align_idx_pairs = []
    for enttype, idx_list in enttype2dataidx.items():
        for idx in idx_list:
            other_idx_list = random.sample(idx_list, args.data_pairs_k)
            for other_idx in other_idx_list:
                align_idx_pairs.append((idx, other_idx))
    # uniformity data pairs # Han: can change to only include negative pairs here
    # ## random sampling pairs
    # uniform_idx_pairs = []
    # idx_list = list(range(len(tokenized_dataset_split)))
    # for idx in idx_list:
    #     other_idx_list = random.sample(idx_list, args.data_pairs_k)
    #     for other_idx in other_idx_list:
    #         uniform_idx_pairs.append((idx, other_idx))
    ## sampling negative pairs only
    uniform_idx_pairs = []
    dict_of_aggregated_other_idx_list = defaultdict(list)
    for enttype, idx_list in enttype2dataidx.items():
        for other_enttype, other_idx_list in enttype2dataidx.items():
            if other_enttype == enttype:
                continue
            dict_of_aggregated_other_idx_list[enttype].extend(other_idx_list)
    for enttype, idx_list in enttype2dataidx.items():
        for idx in idx_list:
            other_idx_list = random.sample(dict_of_aggregated_other_idx_list[enttype], args.data_pairs_k)
            for other_idx in other_idx_list:
                uniform_idx_pairs.append((idx, other_idx))

    field_0, field_1, field_2, field_3, field_4, ofield_0, ofield_1, ofield_2, ofield_3, ofield_4 =  [], [], [], [], [], [], [], [], [], []
    for idx_pair in align_idx_pairs:
        field_0.append(input_ids_without_masks[idx_pair[0]])
        field_1.append(input_ids_with_masks[idx_pair[0]])
        field_2.append(attention_ids[idx_pair[0]])
        field_3.append(recon_labels[idx_pair[0]])
        field_4.append(entity_type[idx_pair[0]])
        ofield_0.append(input_ids_without_masks[idx_pair[1]])
        ofield_1.append(input_ids_with_masks[idx_pair[1]])
        ofield_2.append(attention_ids[idx_pair[1]])
        ofield_3.append(recon_labels[idx_pair[1]])
        ofield_4.append(entity_type[idx_pair[1]])
    align_dataset = TensorDataset(torch.LongTensor(field_0), torch.LongTensor(field_1), torch.LongTensor(field_2), torch.LongTensor(field_3), torch.LongTensor(field_4), torch.LongTensor(ofield_0), torch.LongTensor(ofield_1), torch.LongTensor(ofield_2), torch.LongTensor(ofield_3), torch.LongTensor(ofield_4))
    align_sampler = RandomSampler(align_dataset)
    align_dataloader = DataLoader(align_dataset, sampler=align_sampler, batch_size=batch_size)

    field_0, field_1, field_2, field_3, field_4, ofield_0, ofield_1, ofield_2, ofield_3, ofield_4 =  [], [], [], [], [], [], [], [], [], []
    for idx_pair in uniform_idx_pairs:
        field_0.append(input_ids_without_masks[idx_pair[0]])
        field_1.append(input_ids_with_masks[idx_pair[0]])
        field_2.append(attention_ids[idx_pair[0]])
        field_3.append(recon_labels[idx_pair[0]])
        field_4.append(entity_type[idx_pair[0]])
        ofield_0.append(input_ids_without_masks[idx_pair[1]])
        ofield_1.append(input_ids_with_masks[idx_pair[1]])
        ofield_2.append(attention_ids[idx_pair[1]])
        ofield_3.append(recon_labels[idx_pair[1]])
        ofield_4.append(entity_type[idx_pair[1]])
    uniform_dataset = TensorDataset(torch.LongTensor(field_0), torch.LongTensor(field_1), torch.LongTensor(field_2), torch.LongTensor(field_3), torch.LongTensor(field_4), torch.LongTensor(ofield_0), torch.LongTensor(ofield_1), torch.LongTensor(ofield_2), torch.LongTensor(ofield_3), torch.LongTensor(ofield_4))
    uniform_sampler = RandomSampler(uniform_dataset)
    uniform_dataloader = DataLoader(uniform_dataset, sampler=uniform_sampler, batch_size=batch_size)

    return train_dataloader, align_dataloader, uniform_dataloader


def evaluate_properties(model, args, accelerator, align_dataloader, uniform_dataloader):
    model.eval() # subjective to changes

    null_label_id = torch.LongTensor([-100]) # MLM non-label id is -100, can specify multiple
    null_label_id = null_label_id.to(accelerator.device).view(-1, 1) # utilize the broadcasting feature

    ####
    metric_list = [0, 0, 0, 0] # embedding alignment, embedding uniformity, gradient alignment, gradient uniformity

    # embedding alignment evaluation
    mean_alignment = []
    for step, batch in enumerate(align_dataloader):
        if step >= args.eval_data_cap:
            break
        input_ids_without_masks, input_ids_with_masks, attention_ids, recon_labels, entity_type = batch[:5]
        other_input_ids_without_masks, other_input_ids_with_masks, other_attention_ids, other_recon_labels, other_entity_type = batch[5:]
        batch1 = {'input_ids': input_ids_without_masks, \
            'attention_mask': attention_ids, \
            'labels': recon_labels, \
            }
        batch2 = {'input_ids': other_input_ids_without_masks, \
            'attention_mask': other_attention_ids, \
            'labels': other_recon_labels, \
            }
        assert len(batch1['input_ids']) == 1
        idx_labels_1 = torch.where(batch1['labels'].view(1, -1).ne(null_label_id).sum(0))[0]
        idx_labels_2 = torch.where(batch2['labels'].view(1, -1).ne(null_label_id).sum(0))[0]
        with torch.no_grad():
            # Han: use the last hidden states for interpretation
            outputs1 = model(**batch1, output_hidden_states=True) # NEED DEBUG: currently the model is a sequenceclassification model not LM
            outputs1 = torch.squeeze(outputs1.hidden_states[-1], 0)
            outputs2 = model(**batch2, output_hidden_states=True)
            outputs2 = torch.squeeze(outputs2.hidden_states[-1], 0)
            repr_1, rcount_1 = 0, 0
            repr_2, rcount_2 = 0, 0
            for idx_of_loss in idx_labels_1:
                repr_1 = repr_1 + outputs1[idx_of_loss].detach()
                rcount_1 += 1
            for idx_of_loss in idx_labels_2:
                repr_2 = repr_2 + outputs2[idx_of_loss].detach()
                rcount_2 += 1
            repr_1 = repr_1 / rcount_1 # mean pooling
            repr_2 = repr_2 / rcount_2
            mean_alignment.append(align_property(repr_1, repr_2).item())
    metric_list[0] = np.mean(mean_alignment)

    # embedding uniformity evaluation
    mean_uniformity = []
    for step, batch in enumerate(uniform_dataloader):
        if step >= args.eval_data_cap:
            break
        input_ids_without_masks, input_ids_with_masks, attention_ids, recon_labels, entity_type = batch[:5]
        other_input_ids_without_masks, other_input_ids_with_masks, other_attention_ids, other_recon_labels, other_entity_type = batch[5:]
        batch1 = {'input_ids': input_ids_without_masks, \
            'attention_mask': attention_ids, \
            'labels': recon_labels, \
            }
        batch2 = {'input_ids': other_input_ids_without_masks, \
            'attention_mask': other_attention_ids, \
            'labels': other_recon_labels, \
            }
        assert len(batch1['input_ids']) == 1
        idx_labels_1 = torch.where(batch1['labels'].view(1, -1).ne(null_label_id).sum(0))[0]
        idx_labels_2 = torch.where(batch2['labels'].view(1, -1).ne(null_label_id).sum(0))[0]
        with torch.no_grad():
            # Han: use the last hidden states for interpretation
            outputs1 = model(**batch1, output_hidden_states=True)
            outputs1 = torch.squeeze(outputs1.hidden_states[-1], 0)
            outputs2 = model(**batch2, output_hidden_states=True)
            outputs2 = torch.squeeze(outputs2.hidden_states[-1], 0)
            repr_1, rcount_1 = 0, 0
            repr_2, rcount_2 = 0, 0
            for idx_of_loss in idx_labels_1:
                repr_1 = repr_1 + outputs1[idx_of_loss].detach()
                rcount_1 += 1
            for idx_of_loss in idx_labels_2:
                repr_2 = repr_2 + outputs2[idx_of_loss].detach()
                rcount_2 += 1
            repr_1 = repr_1 / rcount_1 # mean pooling
            repr_2 = repr_2 / rcount_2
            mean_uniformity.append(uniform_property(repr_1, repr_2).item())
    metric_list[1] = np.log(np.mean(mean_uniformity))

    ####

    # gradient alignment evaluation
    mean_alignment = []
    for step, batch in enumerate(align_dataloader):
        if step >= args.eval_data_cap:
            break
        input_ids_without_masks, input_ids_with_masks, attention_ids, recon_labels, entity_type = batch[:5]
        other_input_ids_without_masks, other_input_ids_with_masks, other_attention_ids, other_recon_labels, other_entity_type = batch[5:]
        batch1 = {'input_ids': input_ids_with_masks, \
            'attention_mask': attention_ids, \
            'labels': recon_labels, \
            }
        batch2 = {'input_ids': other_input_ids_with_masks, \
            'attention_mask': other_attention_ids, \
            'labels': other_recon_labels, \
            }
        assert len(batch1['input_ids']) == 1
        # idx_labels_1 = torch.where(batch1['labels'].view(1, -1).ne(null_label_id).sum(0))[0]
        # idx_labels_2 = torch.where(batch2['labels'].view(1, -1).ne(null_label_id).sum(0))[0]
        outputs = model(**batch1)
        loss = outputs.loss
        grads = get_grads(loss, model)
        repr_1 = gather_flat_grads(grads)
        outputs = model(**batch2)
        loss = outputs.loss
        grads = get_grads(loss, model)
        repr_2 = gather_flat_grads(grads)
        mean_alignment.append(align_property(repr_1, repr_2).item())
    metric_list[2] = np.mean(mean_alignment)

    # gradient uniformity evaluation
    mean_uniformity = []
    for step, batch in enumerate(uniform_dataloader):
        if step >= args.eval_data_cap:
            break
        input_ids_without_masks, input_ids_with_masks, attention_ids, recon_labels, entity_type = batch[:5]
        other_input_ids_without_masks, other_input_ids_with_masks, other_attention_ids, other_recon_labels, other_entity_type = batch[5:]
        batch1 = {'input_ids': input_ids_with_masks, \
            'attention_mask': attention_ids, \
            'labels': recon_labels, \
            }
        batch2 = {'input_ids': other_input_ids_with_masks, \
            'attention_mask': other_attention_ids, \
            'labels': other_recon_labels, \
            }
        assert len(batch1['input_ids']) == 1
        # idx_labels_1 = torch.where(batch1['labels'].view(1, -1).ne(null_label_id).sum(0))[0]
        # idx_labels_2 = torch.where(batch2['labels'].view(1, -1).ne(null_label_id).sum(0))[0]
        outputs = model(**batch1)
        loss = outputs.loss
        grads = get_grads(loss, model)
        repr_1 = gather_flat_grads(grads)
        outputs = model(**batch2)
        loss = outputs.loss
        grads = get_grads(loss, model)
        repr_2 = gather_flat_grads(grads)
        mean_uniformity.append(uniform_property(repr_1, repr_2).item())
    metric_list[3] = np.log(np.mean(mean_uniformity))

    ####
    metric_tensor = torch.FloatTensor(metric_list).unsqueeze(0).to(accelerator.device)
    gathered_metric_tensor = accelerator.gather(metric_tensor)
    if accelerator.is_main_process:
        # can save results to file
        metric_report_list = list(gathered_metric_tensor.mean(dim=0).detach().clone().cpu().numpy())
        args.logger.info(metric_report_list)
        with open(os.path.join(args.output_dir, 'evaluated_properties.txt'), 'w') as text_file:
            text_file.write(f"Alignment using embedding representation: {metric_report_list[0]}\n")
            text_file.write(f"Uniformity using embedding representation: {metric_report_list[1]}\n")
            text_file.write(f"Alignment using gradient representation: {metric_report_list[2]}\n")
            text_file.write(f"Uniformity using gradient representation: {metric_report_list[3]}\n")
    accelerator.wait_for_everyone()
