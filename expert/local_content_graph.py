import pandas as pd
import json
import sys,os
import argparse
from argparse import Namespace
from itertools import combinations, product
import numpy as np
import networkx as nx

import graphviz
from graphviz import Digraph

import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex

from expert import nlp_pipeline as pl
from expert.alignment_utils import *

config_text = """[1]
model_type = spert
model_path = {spert_path}/data/models/scierc
tokenizer_path = {spert_path}/data/models/scierc
dataset_path = {spert_path}/data/datasets/local_graphs/temp.json
types_path = {spert_path}/data/datasets/scierc/scierc_types.json
predictions_path = {spert_path}/data/datasets/local_graphs/predictions_temp.json
spacy_model = en_core_web_sm
eval_batch_size = 1
rel_filter_threshold = 0.4
size_embedding = 25
prop_drop = 0.1
max_span_size = 10
sampling_processes = 4
max_pairs = 1000"""


class localContentGraph:
    """
    Local content graph generation and visualization
    """
    def __init__(self,nlp_file=None,ner_tag_file=None,json_file=None,text=None,
                 spert_path='./spert/'):

        if not text is None:
            if not os.path.isdir(spert_path):
                raise ValueError('SPERT path is not valid.  SPERT is required to process text input.')
        
        if nlp_file is None and ner_tag_file is None and text is None:
            raise ValueError("Must provide at least one of nlp_file and ner_tag_file")

        self.spert_path = spert_path
        self.colors = {"Task": "#e41a1c", 
                       "Method": "#377eb8", 
                       "Material": "#4daf4a", 
                       "Metric": "#a65628",
                       "OtherScientificTerm": "#984ea3", 
                       "Generic": "#ff7f00",
                       'Process':"#377eb8"}

        self.nlp_df = None
        self.srl_df = None
        self.ner_preds = None
        self.ner_edges = None
        self.ner_nodes = None
        self.status = True
        self.align_df = None
        
        if not text is None:

            with open(f'{self.spert_path}/configs/local_graph.conf','w') as f:
                f.write(config_text.format(spert_path=spert_path))
                
            self.run_spert(text)
            self.ner_preds = self.load_ner_model_predictions(f'{spert_path}/data/datasets/local_graphs/predictions_temp.json')

            pipe = pl.Pipeline(data_fn=text,
                               verbose=True, batch_size=100, 
                               enable_save=False, overwrite_dir=False)

            functions_to_run = ['coref', 'srl','np']
            pipe.run_nlp_discovery(order=functions_to_run)
            pipe.combine_predictions_to_tuples(subset=functions_to_run)

            self.nlp_df = pipe.result_df
            
        if not nlp_file is None:
            self.nlp_df = pd.read_json(nlp_file,lines=True)

        if not self.nlp_df is None:
            if 'section' in self.nlp_df.columns:
                self.nlp_df['sentence_id'] = self.nlp_df['section'] + '_' + self.nlp_df['sentence_id']

            self.nlp_df['token_id'] = range(len(self.nlp_df))
            self.nlp_df['token_id'] = self.nlp_df['token_id'].astype(str)
            
            self.coref_df = self.parse_coref()
            self.srl_df = self.parse_srl()
            if self.srl_df is None:
                self.status=False
                return None
            for col in ['ARG0','ARG1','V']:
                if col not in self.srl_df.columns:
                    self.status = False
                    return None
                
            self.filter_srls()
            #self.clean_srls()

            self.combine_srl_coref()
            
        if not ner_tag_file is None and text is None and os.path.isfile(ner_tag_file):
            self.ner_preds = self.load_ner_model_predictions(ner_tag_file)
            token_count = 0
            for pred in self.ner_preds:
                pred['nlp_token_map'] = [str(token_count + i) for i in range(len(pred['tokens']))]
                token_count = token_count + len(pred['tokens'])
                
        elif not ner_tag_file is None and not os.path.isfile(ner_tag_file):
            ner_tag_file = None

        if not self.nlp_df is None and not self.ner_preds is None:
            self.align_tokens()

        if not self.ner_preds is None:
            self.extract_relations()


        self.combined_edges = None
        self.combined_nodes = None
        if (not ner_tag_file is None and not nlp_file is None) or not text is None:
            self.combine_graphs()

    def run_spert(self,text):
        sys.path.append(os.path.realpath(self.spert_path))
            
        from args import train_argparser, eval_argparser, predict_argparser
        from config_reader import process_configs, _yield_configs
            
        self.process_data_for_spert(text)

        sys.argv.extend(['--config', f'{self.spert_path}/configs/local_graph.conf'])
        arg_parser = predict_argparser()

        args, _ = arg_parser.parse_known_args()

        for run_args, run_config, run_repeat in _yield_configs(arg_parser,args):
            self.predict_spert(run_args)
    
            
    def predict_spert(self,run_args):

        from spert.spert_trainer import SpERTTrainer
        from spert import input_reader

        trainer = SpERTTrainer(run_args)
        trainer.predict(dataset_path=run_args.dataset_path, types_path=run_args.types_path,
                        input_reader_cls=input_reader.JsonPredictionInputReader)
            
    def process_data_for_spert(self,text):

        print('Processing data for SPERT...')
        
        nlp = spacy.load('en_core_web_trf')

        sentences = [sent.text.strip() for sent in nlp(text).sents if len(sent.text.strip()) > 50]
        
        infixes = nlp.Defaults.prefixes + [r"[./]", r"[-]~", r"(.'.)"]
        infix_re = spacy.util.compile_infix_regex(infixes)

        def custom_tokenizer(nlp):
            return Tokenizer(nlp.vocab, infix_finditer=infix_re.finditer)

        nlp.tokenizer = custom_tokenizer(nlp)

        output = []
        for sent in sentences:
            doc = nlp(sent)
            tokens = [token.text for token in doc]

            output.append({'text':sent,
                           'tokens':tokens})


        out = f'{self.spert_path}/data/datasets/local_graphs/temp.json'
        out_path = '/'.join(out.split('/')[:-1])
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        with open(out,'w') as f:
            json.dump(output,f)
            
    def generate_graph(self,dataframe, source_node_column_name,
                       target_node_column_name, edge_label_column_name,
                       *, nodes=None, engine='dot', weighted=True, scale_vis_edge_weight=1.0, 
                       max_edge_weight=None, color_edge_over_max='red', rankdir="LR"):
        """

        e.g. for generating a graph that encodes ARG0 --Verb--> ARG1:
        source_node_column_name = 'ARG0'
        target_node_column_name = 'ARG1'
        edge_label_column_name = 'Verb'

        :param dataframe: pandas dataframe that contains the
        relationships to include in the directed graph.
        :param source_node_column_name: (string) column name in
        the dataframe that includes the source node in the edge per row
        :param target_node_column_name: (string) column name in
        the dataframe that includes the target node in the edge per row
        :param edge_label_column_name: (string) column name in
        the dataframe that includes the label for the edge per row
        :param engine: (string) specifies the engine to use for graphviz,
        default is 'sfdp' but can also use 'neato','dot','circo'

        :param weighted: (boolean) flag that indicates whether to collapse edges and represent frequency with
        edge thickness
        :param scale_vis_edge_weight: (float) scale factor to apply to edge weight thickness,
         by default = 1.0 to use raw frequency as thickness

        :return: graphviz DiGraph
        """

        if weighted:
            weight_col = 'weight'
            dataframe[weight_col] = 1
            dataframe = dataframe.groupby([source_node_column_name,
                                      target_node_column_name,
                                      edge_label_column_name],as_index=False)['weight'].sum()

        g = Digraph('G', engine=engine, graph_attr = {"rankdir":rankdir})

        if not nodes is None:
            for i in nodes.index:
                g.node(nodes.loc[i,'entity'],
                       color=nodes.loc[i,'color'],
                      fontcolor=nodes.loc[i,'color'])

        for i in dataframe.index:
            if weighted:
                if max_edge_weight is not None and dataframe.loc[i, weight_col] > max_edge_weight:

                    g.edge(dataframe.loc[i,source_node_column_name],
                           dataframe.loc[i, target_node_column_name],
                           label=dataframe.loc[i, edge_label_column_name],
                           penwidth = str(max_edge_weight * scale_vis_edge_weight),
                           color = color_edge_over_max) 
                else:
                    g.edge(dataframe.loc[i,source_node_column_name],
                           dataframe.loc[i, target_node_column_name],
                           label=dataframe.loc[i, edge_label_column_name],
                           penwidth = str(dataframe.loc[i, weight_col] * scale_vis_edge_weight))
            else:
                g.edge(dataframe.loc[i,source_node_column_name],
                       dataframe.loc[i, target_node_column_name],
                       label=dataframe.loc[i, edge_label_column_name])

        return g



    def visualize(self,graphviz_graph):
        """
        Save graph visualization to pdf
        :param graphviz_graph: graphviz object for the casual graph
        """
        # display graphviz visualization of causal graph
        src_of_G = graphviz.Source(graphviz_graph)
        display(src_of_G)


    def align_tokens(self):

        nlp_tokens =self.nlp_df[['token','token_id']]['token'].tolist()

        ner_tokens = []
        sentences = []
        for s, sent in enumerate(self.ner_preds):
            ner_tokens += sent['tokens']
            sentences += [s]*len(sent['tokens'])

        align1, align2, align1_idx, align2_idx = needleman_wunsch(nlp_tokens, ner_tokens)
        self.align_df = pd.DataFrame({'nlp_token':align1,'ner_token':align2,
                                 'nlp_idx':align1_idx,'ner_idx':align2_idx})

        sentence_df = pd.DataFrame({'sentence':sentences})
        sentence_df['ner_idx'] = range(len(sentence_df))

        self.align_df = self.align_df.merge(sentence_df,on='ner_idx',how='left')

        self.align_df['ner_token'] = self.align_df['ner_token'].replace({'[-]':np.nan}).bfill()
        self.align_df['ner_idx'] = self.align_df['ner_idx'].replace({-1:np.nan}).bfill()
        self.align_df['sentence'] = self.align_df['sentence'].replace({-1:np.nan}).bfill()

        grps = self.align_df.groupby(['ner_token',
                                 'ner_idx',
                                 'sentence'])['nlp_idx']
        align_map = grps.apply(lambda x: '-'.join([str(n) for n in x])).reset_index().sort_values('ner_idx')

        token_lists = []
        for g,grp in align_map.groupby('sentence'):
            token_lists.append(list(grp['nlp_idx'].values))
            self.ner_preds[int(g)]['nlp_token_map'] = list(grp['nlp_idx'].values)

            
    def load_ner_model_predictions(self,ner_tag_file):

        with open(ner_tag_file,'r') as f:
            ner_preds = json.load(f)

        return(ner_preds)


    def parse_sentence_srl(self, sentence_df):
        
        if not isinstance(sentence_df.iloc[0]['srl_args'],dict):
            return

        srls = sentence_df['srl_args'].apply(pd.Series).fillna('O')
        srl_cols = srls.columns
        sentence_df = pd.concat([sentence_df,srls],axis=1)

        arg_dfs = []
        for col in srl_cols:

            sentence_df[col] = sentence_df[col].str.replace('B-','').str.replace('I-','')


            args = {}
            for arg_type in sentence_df[col].unique():


                if arg_type == 'O':
                    continue
                subset = sentence_df[sentence_df[col] == arg_type]
                nps = [x for x in list(subset['noun phrases'].unique()) if x != '']

                arg = ' '.join(subset['token'])
                arg_id = '-'.join(subset['token_id'])

                args[arg_type] = [arg]
                args[f'{arg_type}_np'] = [nps]
                args[f'{arg_type}_ids'] = [arg_id]

                
            args = pd.DataFrame(args)

            arg_dfs.append(args)

        if len(arg_dfs) > 0:
            df = pd.concat(arg_dfs)
            df['doc_id'] = sentence_df['doc_id'].values[0]
            df['sentence_id'] = sentence_df['sentence_id'].values[0]
            df['sentence'] = ' '.join(sentence_df['token'])

            return(df)

    def strictly_increasing(self,L):
        return all(x<y for x, y in zip(L, L[1:]))

    def contains(self,small, big):
        for i in range(len(big)-len(small)+1):
            for j in range(len(small)):
                if big[i+j] != small[j]:
                    break
                else:
                    return i, i+len(small)
        return False
    
    def assign_coref_token_ids(self,grp):
        id_list = []
        cluster = grp['cluster'].values[0]
        for ref in cluster:
            id_str = ''
            words = ref.split('-')

            while self.contains(["",""],words):
                idx = self.contains(["",""],words)
                words = words[:idx[0]] + ["-"] + words[idx[1]:]
                
            w_idx_list = []
            for w in words:
                w_idx_list.append(list(grp[grp['token'] == w]['token_id'].values))
            possible_tokens = list(product(*w_idx_list))
            
            filtered_options = []
            token_ranges = []
            for l in possible_tokens:
                int_tokens = [int(t) for t in l]
                if self.strictly_increasing(int_tokens):
                    filtered_options.append(l)
                    token_ranges.append(max(int_tokens) - min(int_tokens))

            if len(token_ranges) > 0:
                token_ranges = np.array(token_ranges)
                idx = np.where(token_ranges == np.min(token_ranges))[0]
            else:
                idx = []
                
            for i in idx:
                id_list.append('-'.join(filtered_options[i]))
                    
       
        return(id_list)
        
    def parse_coref(self):

        if not 'cluster' in self.nlp_df.columns:
            print('No coref in NLP output')
            empty_df = pd.DataFrame(columns=['from','to','label'])
            return(empty_df)
        
        temp = self.nlp_df[self.nlp_df['cluster'].str.len() > 0].explode('cluster')
        temp['cluster_str'] = [','.join(map(str, l)) for l in temp['cluster']]

        cluster2token = {}
        for g,grp in temp.groupby('cluster_str'):
            token_ids = self.assign_coref_token_ids(grp)
            cluster2token[g] = token_ids
            
        temp = temp.drop_duplicates('cluster_str')
        temp = temp[temp['cluster'].str.len() > 1]
        
        corefs = []
        for clust in temp['cluster_str']:
            clust = cluster2token[clust]
            coref_edges = pd.DataFrame(combinations(clust,2))
            coref_edges.columns = ['from','to']
            coref_edges['label'] = 'coref'
            corefs.append(coref_edges)
        coref_edges = pd.concat(corefs)

        return(coref_edges)
    
    def parse_srl(self):
        
        srl_dfs = []
        for g,grp in self.nlp_df.groupby('sentence_id'):
            srl_df = self.parse_sentence_srl(grp)
            if not srl_df is None:
                srl_df = srl_df.reset_index(drop=True)
                srl_dfs.append(srl_df)
        if len(srl_dfs) > 0:
            srl_df = pd.concat(srl_dfs)
        else:
            srl_df = None

        return(srl_df)


    def filter_srls(self):

        self.srl_df = self.srl_df.dropna(subset=[c for c in self.srl_df.columns if c in ['ARG0','ARG1','V']])

        idx = self.srl_df['ARG0_np'].apply(lambda x: len(x)) == 1
        idx = idx & (self.srl_df['ARG1_np'].apply(lambda x: len(x)) == 1)

        self.srl_df = self.srl_df[idx]

        self.srl_df['ARG0_np'] = self.srl_df['ARG0_np'].apply(lambda x: x[0]).str.lower()
        self.srl_df['ARG1_np'] = self.srl_df['ARG1_np'].apply(lambda x: x[0]).str.lower()


    def clean_articles(self,x,article_list=['the','a','our','this','and','these']):

        for word in article_list:
            if x.lower().startswith(f"{word} "):
                x = x[len(f"{word} "):]

        return(x.lower())

    def clean_srls(self):

        self.srl_df['ARG0'] = self.srl_df['ARG0'].apply(self.clean_articles)
        self.srl_df['ARG1'] = self.srl_df['ARG1'].apply(self.clean_articles)
        self.srl_df['ARG0_np'] = self.srl_df['ARG0_np'].apply(self.clean_articles)
        self.srl_df['ARG1_np'] = self.srl_df['ARG1_np'].apply(self.clean_articles)


    def combine_srl_coref(self):

        #self.srl_df = self.srl_df[['ARG0_np','V','ARG1_np']].copy()
        self.srl_df = self.srl_df[['ARG0_ids','V','ARG1_ids','ARG0_np','ARG1_np']].copy()
        self.srl_df.columns = ['from','label','to','from_label','to_label']
        self.srl_df = pd.concat([self.srl_df,self.coref_df])
        self.srl_df['weight'] = 1

    def merge_coref(self,df):
        
        to_map = df.set_index('to').dropna(subset=['to_label']).to_dict()['to_label']
        from_map = df.set_index('from').dropna(subset=['from_label']).to_dict()['from_label']
        label_map = dict(to_map,**from_map)
        
        G=nx.from_pandas_edgelist(df[df['label'] == 'coref'], 'from', 'to')

        for subgraph in list(nx.connected_components(G)):
            subgraph_df = df[ (df['to'].isin(subgraph)) | (df['from'].isin(subgraph)) ]
            if (subgraph_df['label'] == 'coref').sum() > 0 and (subgraph_df['label'] != 'coref').sum() > 0:
                for clique in list(nx.find_cliques(G.subgraph(subgraph).copy())):
                    inds = np.array([int(c.split('-')[0]) for c in clique])
                    
                    for i in range(len(inds)):
                        if i != np.argmin(inds):
                            df['from'] = df['from'].replace({clique[i]:clique[np.argmin(inds)]})
                            df['to'] = df['to'].replace({clique[i]:clique[np.argmin(inds)]})                            

        df = df[df['label'] != 'coref'].copy()
        df['from'] = df['from'].replace(label_map)
        df['to'] = df['to'].replace(label_map)

        return(df)
        
    def srl_graph(self,show=True,collapse_coref=False,**kwargs):

        if not self.srl_df is None and not len(self.srl_df) == 0:

            srl_df_out = self.srl_df.copy()

            if not self.align_df is None:
                srl_df_out['to_label'] = srl_df_out['to'].apply(self.get_label_from_tokens).apply(self.clean_articles)
                srl_df_out['from_label'] = srl_df_out['from'].apply(self.get_label_from_tokens).apply(self.clean_articles)

            if collapse_coref:
                    
                srl_df_out = srl_df_out.groupby(['from','to','label',
                                                 'to_label','from_label'],
                                                dropna=False)['weight'].sum().reset_index()        
        
                srl_df_out = self.merge_coref(srl_df_out)
            else:
                srl_df_out['from'] = srl_df_out['from_label']
                srl_df_out['to'] = srl_df_out['to_label']

                
            self.g_srl = self.generate_graph(srl_df_out,'from','to','label',
                                    **kwargs)
            if show:
                self.visualize(self.g_srl)
        elif self.srl_df is None:
            print('There is no SRL data')
        else:
            print('There are no edges in the SRL data')

    def extract_sentence_relations(self,pred):
        
        entities = []
        indices = []
        types = []
        for entity in pred['entities']:
            label = entity['type']
            ent = ' '.join(pred['tokens'][entity['start']:entity['end']])

            entities.append(ent)
            types.append(label)
            indices.append('-'.join([pred['nlp_token_map'][x] for x in range(entity['start'],entity['end'])]))

        ntokens = len(pred['tokens'])
            
        nodes = pd.DataFrame({'entity':entities,'type':types,'indices':indices})
        nodes['entity'] = nodes['entity'].astype(str).str.lower()

        froms = []
        tos = []
        from_labels = []
        to_labels = []
        labels = []
        for relation in pred['relations']:
            from_labels.append(nodes['entity'].iloc[relation['head']])
            to_labels.append(nodes['entity'].iloc[relation['tail']])
            froms.append(nodes['indices'].iloc[relation['head']])
            tos.append(nodes['indices'].iloc[relation['tail']])
            labels.append(relation['type'])

        edges = pd.DataFrame({'from':froms,'to':tos,'label':labels,'from_label':from_labels,'to_label':to_labels})
        edges = edges[edges['label'] != 'Conjunction']

        nodes = nodes[ (nodes['entity'].isin(edges['from_label'].values)) | (nodes['entity'].isin(edges['to_label'].values)) ]

        if len(nodes) > 0:
            nodes = nodes[~nodes['entity'].str.isspace()]
        if len(edges) > 0:
            edges = edges[~edges['from'].str.isspace()]
            edges = edges[~edges['to'].str.isspace()]

        return(nodes,edges)


    def extract_relations(self):

        all_nodes = []
        all_edges = []
        for p, pred in enumerate(self.ner_preds):
            nodes, edges = self.extract_sentence_relations(pred)
            nodes['color'] = nodes['type'].map(self.colors)
            all_nodes.append(nodes)
            all_edges.append(edges)
            
        self.ner_nodes = pd.concat(all_nodes).reset_index(drop=True)
        self.ner_edges = pd.concat(all_edges).reset_index(drop=True)
        self.ner_edges['weight'] = 1.0
        self.ner_edges = self.ner_edges.groupby(['from','to','label','from_label','to_label'])['weight'].sum().reset_index()


    def ner_graph(self,show=True,**kwargs):

        if not self.ner_edges is None and not len(self.ner_edges) == 0:

            self.g_ner = self.generate_graph(self.ner_edges,'from_label','to_label','label',
                                    **kwargs,
                                    nodes=self.ner_nodes)
            if show:
                self.visualize(self.g_ner)
        elif self.ner_edges is None: 
            print('There is no NER data')
        else:
            print('There are no edges in the NER data')

    def get_label_from_tokens(self,x):

        tokens = x.split('-')

        words = self.align_df.loc[self.align_df['nlp_idx'].isin(tokens),'nlp_token']

        return ' '.join(list(words))
            
    def combine_graphs(self):

        srl_filt_nodes = pd.DataFrame(pd.concat([self.srl_df['from'],self.srl_df['to']]).reset_index(drop=True).drop_duplicates())
        srl_filt_nodes.columns = ['indices']
        srl_filt_nodes['type'] = 'Noun Phrase'
        srl_filt_nodes['color'] = 'grey'
        srl_filt_nodes = srl_filt_nodes[~srl_filt_nodes['indices'].isin(self.ner_nodes['indices'].values)]

        self.combined_nodes = pd.concat([self.ner_nodes[['indices','type','color']],srl_filt_nodes]).dropna().reset_index(drop=True)

        self.combined_edges = pd.concat([self.ner_edges,self.srl_df]).reset_index(drop=True)

        self.combined_edges['to_label'] = self.combined_edges['to'].apply(self.get_label_from_tokens).apply(self.clean_articles)
        self.combined_edges['from_label'] = self.combined_edges['from'].apply(self.get_label_from_tokens).apply(self.clean_articles)
        
        self.combined_edges = self.combined_edges.merge(self.combined_nodes,left_on='from',right_on='indices')
        self.combined_edges = self.combined_edges.merge(self.combined_nodes,left_on='to',right_on='indices',suffixes=('_from','_to'))
        
        self.combined_edges = self.combined_edges.groupby(['from','to','label',
                                                           'to_label','from_label',
                                                           'color_to','color_from'],
                                                          dropna=False)['weight'].sum().reset_index()       
        
        self.combined_edges = self.merge_coref(self.combined_edges)

        nodes_from = self.combined_edges[['from','color_from']]
        nodes_from.columns = ['entity','color']
        nodes_to = self.combined_edges[['to','color_to']]
        nodes_to.columns = ['entity','color']
        
        self.combined_nodes = pd.concat([nodes_from,nodes_to]).drop_duplicates().reset_index(drop=True)


        self.combined_edges = self.combined_edges.drop(['to_label','from_label','color_to','color_from'],axis=1)

    def combined_graph(self,show=True,**kwargs):

        if not self.combined_edges is None:
            self.g_combined = self.generate_graph(self.combined_edges,'from','to','label',
                                    **kwargs,
                                    nodes=self.combined_nodes)
            if show:
                self.visualize(self.g_combined)
        else:
            print('There is no combined graph. Need both NER and SRL data.')

    def save_graph(self,fn_id,path='./'):

        if not self.combined_edges is None:
            self.combined_edges.to_json(f'{path}/{fn_id}_local_graph_combined_edges.jsonl',lines=True,orient='records')
            self.combined_nodes.to_json(f'{path}/{fn_id}_local_graph_combined_nodes.jsonl',lines=True,orient='records')      
            try:
                self.g_combined.render(filename=f'{path}/{fn_id}_local_graph_combined')
            except:
                ''
        elif not self.ner_edges is None:
            self.ner_edges.to_json(f'{path}/{fn_id}_local_graph_ner_edges.jsonl',lines=True,orient='records')
            self.ner_nodes.to_json(f'{path}/{fn_id}_local_graph_ner_nodes.jsonl',lines=True,orient='records')
            try:
                self.g_ner.render(filename=f'{path}/{fn_id}_local_graph_ner')
            except:
                ''
        elif not self.srl_df is None:
            self.srl_df.to_json(f'{path}/{fn_id}_local_graph_srl_edges.jsonl',lines=True,orient='records')
            #self.srl_nodes.to_json(f'{path}/{fn_id}_local_graph_srl_nodes.jsonl',lines=True,orient='records')
            try:
                self.g_srl.render(filename=f'{path}/{fn_id}_local_graph_srl')
            except:
                ''
                
            
