import os
import sys
import pickle
import json
import pandas as pd
import time
import numpy as np
import torch
import argparse
from transformers import AutoModel, AutoTokenizer, AutoConfig, BertModel
import re


def BERT_vectorize_averaged(args, text, batch_size=128, device="cpu"):
    """
    Use the average of the token embeddings in the second to last layer of the model as features

    Parameters:
        text: list, numpy array, or pandas series of strings
        args: one of the huggingface transformer models (e.g. "bert-base-cased", "distilbert-base-cased")
        batch_size: int
    Returns:
        numpy array of features with shape (len(text),hidden_size)
    """

    # Instantiate the tokenizer and model
    model_path = args.model + 'pytorch_model.bin'

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_config = AutoConfig.from_pretrained(args.model, output_hidden_states=True)
    model_state_dict = torch.load(model_path)
    model = AutoModel.from_pretrained(args.bert, state_dict=model_state_dict, config=model_config).to(device)

    features, padded_text = [], []
    i = 0
    words = []
    while i < len(text):
        # batch the model inference to fit in memory
        end = min(i + batch_size, len(text))
        batch = text[i:end]

        tokenized_batch = tokenizer(batch,
                                    max_length=512,
                                    padding="max_length",
                                    add_special_tokens=True,
                                    return_attention_mask=True,
                                    return_tensors="pt",
                                    truncation=True)
        # print(tokenized_batch["input_ids"].numpy().shape)
        # Must decode word for word; inconsistent array length given when decoding an entire sequence
        for t in tokenized_batch["input_ids"].numpy():
            for i in t:
                words.append(tokenizer.decode([i]))

        with torch.no_grad():
            batch_outputs = model(input_ids=tokenized_batch["input_ids"].to(device),
                                  attention_mask=tokenized_batch["attention_mask"].to(device))
        # Get the tuple of all hidden states
        if "distilbert" in args.bert:
            # distilbert returns a different shape output
            batch_hidden_states = batch_outputs[1]
        else:
            batch_hidden_states = batch_outputs[2]

        # specific hidden state for the second to last layer
        batch_hidden_layer = batch_hidden_states[-1]
        # (16, 512, 768)
        # print('batch_hidden_layer', batch_hidden_layer.shape)
        batch_hidden_layer = batch_hidden_layer.cpu().numpy()
        features.append(batch_hidden_layer)

        i = end
        print(f"{i}/{len(text)}") 
    return np.concatenate(features), words



def cleanEmbeddingDataFrame(df):
    # remove padding tokens
    embToks = ['[CLS]','[PAD]','[SEP]']  
    df= df[~df['token'].isin(embToks)].copy().reset_index(drop=False)
        
    keyCol = 'gbkey' 
    
    def indexKey(x):
        if x['token'][:2] == '##':
            return np.nan
        else:
            return x['index']
    
    mask = [x[:2]=='##' for x in df['token']]
    df['index'][mask] = np.nan 
    df['index'] = df['index'].fillna(method='ffill')
    keyCol = 'index'
        
    gbCols = [keyCol, 'nodeID', 'country', 'date']
    embCols = set(df.columns) - set(gbCols+['token'])
    transform = {'token': lambda x: ''.join(x).replace('##','')}
    transform.update({str(x):np.mean for x in range(768)})
     
    df = df.groupby(gbCols).agg(transform).reset_index().drop(columns=[keyCol])
     
    #remove numeric or punctuation only tokens
    def charOnly(x):
        return re.sub(' +', ' ', re.sub('[\W0-9]',' ', x)).strip()
    
    df= df[df['token'].apply(lambda x:len(charOnly(x))>0)].copy()
        
    return df



def extractESTEEMembeddings(arguments):
    """
    Arguments via command line:
    
    parser.add_argument('-bert', type=str, default='bert-base-uncased',
                        help='Name of bert model used for fine-tuning.')
    parser.add_argument('-model', type=str, help='path to parent_dir of pytorch_model.bin')
    parser.add_argument('-outdir', type=str, help='path to dir to save embeddings')
    parser.add_argument('-data', type=str, help='.json or jsonl file')
    """
    df = pd.read_json(arguments.data, lines=True, orient='records')
    df['human date'] = pd.to_datetime(df['date'], unit='s')
    df['month'] = df['human date'].dt.month
    # df = df.loc[df['nodeID'] == 'Scopus-2-s2.0-85084659242']
    for j, grp in df.groupby('month'):
        grp = grp.reset_index(drop=True)
        texts = [x.replace("[BOS]", "").replace("[EOS]", "") for x in list(grp['text'])]
        vecs, pad_text = BERT_vectorize_averaged(arguments, texts, 16)

        embeddings = []
        ids, dates, countries = [], [], []
        for i, vector in enumerate(vecs):
            embeddings.extend(vector)
            idx = grp.iloc[i]['nodeID']
            dat = grp.iloc[i]['date']
            count = grp.iloc[i]['country']
            ids.extend([idx] * len(vector))
            dates.extend([dat] * len(vector))
            countries.extend([count] * len(vector))

        print(len(embeddings), np.asarray(embeddings).shape)

        new_df = pd.DataFrame(data=embeddings, index=pad_text, columns=[x for x in range(len(embeddings[0]))])
        num_columns = list(new_df.columns)
        new_df = new_df.reset_index(drop=False).rename(columns={'index': "token"})
        new_df['nodeID'] = ids
        new_df['country'] = countries
        new_df['date'] = dates
        new_df = new_df[['token', 'nodeID', 'country', 'date'] + num_columns]
        new_df.columns = new_df.columns.astype(str)

        # clean embeddings (remove PAD, SEP, CLS; combine OOV words; remove punctuation/numeric only)
        new_df = cleanEmbeddingDataFrame(new_df)
        
        if not os.path.exists(arguments.outdir):
            os.makedirs(arguments.outdir)

        new_df.to_parquet(arguments.outdir + arguments.bert + "_embeddings_" + str(j) + ".parquet")

