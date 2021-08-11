import json
import pandas as pd 
import numpy as np
import math
import re
import string
import itertools
import time
from more_itertools import chunked

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import networkx as nx
from networkx.algorithms import bipartite
import networkx.algorithms.community as nxcom
from networkx.algorithms import approximation as approx

from Levenshtein import distance as levenshtein_distance, ratio
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import pairwise_distances_chunked
from scipy.spatial.distance import pdist, squareform, cdist
from collections import Counter

import tqdm

from IPython.display import display



def calc_row_idx(k, n):
    return int(math.ceil((1/2.) * (- (-8*k + 4 *n**2 -4*n - 7)**0.5 + 2*n -1) - 1))

def elem_in_i_rows(i, n):
    return i * (n - 1 - i) + (i*(i + 1))//2

def calc_col_idx(k, i, n):
    return int(n - elem_in_i_rows(i + 1, n) + k)


class LSTM(torch.nn.Module) :
    def __init__(self, vocab_size, embedding_dim, hidden_dim) :
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.3)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
        self.sm = nn.Sigmoid()
        #self.sm = nn.LogSoftmax()                                                                                                                

    def forward(self, x, s):
        x = self.embeddings(x)
        x = self.dropout(x)
        x_pack = pack_padded_sequence(x, s, batch_first=True, enforce_sorted=False)
        out_pack, (ht, ct) = self.lstm(x_pack)
        out = self.linear(ht[-1])
        out = self.sm(out)
        return out

class EntityMerger:
    
    def __init__(self, nodes, edges,
                 candidate_thresh = 0.8,
                 blacklist = None,
                 merge_log=None,
                 model=None,
                 vocab=None,
                 config = {'text_field':'displayName',
                           'id_field':'nodeID',
                           'node_type':'Scientist',
                           'steps':{
                               'displayName':['lev','tfidf','parts']
                           }}):
        
        self.nodes = nodes.copy()
        self.edges = edges.copy()
        self.candidate_thresh = candidate_thresh
        
        self.text_field = config['text_field']
        self.id_field = config['id_field']
        self.node_type = config['node_type']
        
        self.steps = config['steps']

        self.name_part_likelihoods = {}
        self.name_connectivities = {}
        for field in config['steps'].keys():
            self.nodes[f'{field}_clean'] = self.nodes[field].apply(self.clean_text)

            if 'likelihood'  in config['steps'][field]:
                l,c = self.name_part_graph(f'{field}_clean')
                self.name_part_likelihoods[f'{field}_clean'] = l
                self.name_connectivities[f'{field}_clean'] = c
            
        self.run_predict = False
        if not model is None and not vocab is None:
            self.run_predict = True
            print(model)
            self.model = LSTM(81,32,128)
            self.model.load_state_dict(torch.load(model))
            self.model.eval()

            with open(vocab,'r') as f:
                self.vocab = json.load(f)

            self.max_len=71
                
        if not blacklist is None and '.csv' in blacklist:
            self.blacklist_fn = blacklist
            try:
                self.blacklist = pd.read_csv(blacklist)
                self.blacklist.columns = ['from','to', 'blacklist']
                temp = self.blacklist.copy()
                temp.columns = ['to','from','blacklist']
                self.blacklist = pd.concat([self.blacklist,temp])
            except:
                self.blacklist = pd.DataFrame(columns=['to','from','blacklist'])
        else:
            self.blacklist_fn = None
            self.blacklist = None


        if not merge_log is None and '.csv' in merge_log:
            self.merge_fn = merge_log
            try:
                self.merge_log = pd.read_csv(merge_log)
                self.merge_log.columns = ['match1','match2','canonical']
            except:
                self.merge_log = pd.DataFrame(columns=['match1','match2','canonical'])
        else:
            self.merge_fn = None
            self.merge_log = None

            
    def print_step_options(self):
        print("lev - Levenshtein distance\n"
               "tfidf - Word-level TFIDF vector cosine distance\n"
               "parts - Match words in the pair of strings to identify lowest levenshtein distance (can account for different word order)")

    def encode_text(self, text):
        tokenized = list(text)
        encoded = np.zeros(self.max_len, dtype=int)
        enc1 = np.array([self.vocab.get(token, self.vocab["UNK"]) for token in tokenized])
        length = min(self.max_len, len(enc1))
        encoded[:length] = enc1[:length]
        return encoded, length

        
    def clean_text(self,x):
        x = str(x)
        x = re.sub(r'[‐᠆﹣－⁃−]+','-',x)
        x = re.sub('\s+',' ',x)
        x = x.lower()
        
        return(x)
        
    def path_weight(self, u, v, d):
        edge_wt = d.get("weight", 1)

        return(1.0 - edge_wt / float(self.max_weight))
        
    def get_path_length(self,node1,node2,hops):
        
        try:


            if hops == 2:

                neighborhood = nx.single_source_shortest_path_length(self.proj, node1, cutoff=2)
                neighborhood1 = [k for k,v in neighborhood.items() if v == 1]
                neighborhood2 = [k for k,v in neighborhood.items() if v == 2]
            
            
                if node2 in neighborhood2 and not node2 in neighborhood1:
                
                    paths = list(nx.all_simple_paths(self.proj, source=node1, target=node2, cutoff=2))
                    paths = [p[1] for p in paths if len(p) == 3]
                    
                    dist = 0.0
                    for p in paths:
                        dist += self.proj.get_edge_data(node1, p)['weight']
                        dist += self.proj.get_edge_data(node2, p)['weight']
                
                else:
                    dist = 0.0
            else:
                dist = self.proj.get_edge_data(node1,node2)['weight']
                
        except:
            dist = 0.0
            
        return(dist)
        
    def get_graph_sims(self):

        #institution merging:
        # colocation graph, 1 hop weight
        # shared staff graph, 1 hop weight

        #location merging:
        # shared institutions graph, 1 hop weight

        #author merging:
        # coauthor graph: sum of 2 hop weights
        # coworker graph, 1 hop weight

        if self.node_type == 'Institution':
            graphs = {'colocation':
                     {'nodes':['Institution','Location'],
                      'hops':1
                     },
                     'shared_staff':
                     {'nodes':['Scientist','Institution'],
                      'hops':1
                     }
                     }
        elif self.node_type == 'Location':
            graphs = {'shared_institutions':
                     {'nodes':['Institution','Location'],
                      'hops':1
                     }
                     }
        elif self.node_type == 'Scientist':
            graphs = {'coauthor':
                     {'nodes':['Scientist','Paper'],
                      'hops':2
                     },
                     'colleague':
                     {'nodes':['Scientist','Institution'],
                      'hops':1
                     }
                     }
        elif self.node_type in ['Source','Journal','Book','Conference']:
            graphs = {'colocation':
                     {'nodes':['Source','Journal','Book','Conference','Paper'],
                      'hops':1
                     }
                     }


        graph_sim_cols = []
        for gt,config in graphs.items():

            #print(config)
            
            sel_edges = self.edges[ (self.edges['fromType'] == config['nodes'][0]) & (self.edges['toType'] == config['nodes'][1]) ]
            sel_nodes = self.nodes[ (self.nodes['nodeID'].isin(sel_edges['from'])) | self.nodes['nodeID'].isin(sel_edges['to']) ]
            
            print(f'Constructing {gt} graph...',end="")
            start = time.time()
            G = nx.Graph()
            i = 0
            for g,grp in sel_nodes.groupby('nodeType'):
                G.add_nodes_from(grp['nodeID'], bipartite=i)
                i += 1
            
                G.add_weighted_edges_from([(row['from'], row['to'], 1) for idx, row in sel_edges.iterrows()], 
                                          weight='weight')
        
            end = time.time()
            print(f'{end-start:.2f} s')
        
            print('Projecting graph...',end="")
            start = time.time()
            self.proj = bipartite.weighted_projected_graph(G,sel_nodes[sel_nodes['nodeType'] == self.node_type]['nodeID'].values )
            end = time.time()
            print(f'{end-start:.2f} s')
        
                
            print('Getting path lengths...',end="")
            start = time.time()
            self.text_sims[f'path_weights_{gt}'] = self.text_sims.apply(lambda row: self.get_path_length(row['from'], 
                                                                                                       row['to'],
                                                                                                       config['hops']),
                                                                      axis=1)
            end = time.time()
            print(f'{end-start:.2f} s')

            min_weight = self.text_sims[f'path_weights_{gt}'].min()
            max_weight = self.text_sims[f'path_weights_{gt}'].max()

            if max_weight != 0.0:
                zeros = self.text_sims[self.text_sims[f'path_weights_{gt}'] == 0.0].copy()
                zeros[f'graph_sim_{gt}'] = 0.0
            
                nonzeros = self.text_sims[self.text_sims[f'path_weights_{gt}'] > 0.0].copy()
                nonzeros[f'graph_sim_{gt}'] = nonzeros[f'path_weights_{gt}'].rank(pct=True,
                                                                                  method='max')

                self.text_sims = pd.concat([zeros,nonzeros])
            
            if (max_weight == min_weight):
                self.text_sims[f'graph_sim_{gt}'] = 0.0

            graph_sim_cols.append(f'graph_sim_{gt}')
            
            print()
                
        self.text_sims['graph_sim_total'] = self.text_sims[graph_sim_cols].sum(axis=1)
                
        self.text_sims['sim_total'] = self.text_sims['text_sim_total'] + self.text_sims['graph_sim_total']
        
        return(self.text_sims)      
        
    def pairwise_metric(self, metric_matrix, labels, metric_name, thresh):

        n = len(labels)
        from_nodes = [calc_row_idx(k, n) for i,k in enumerate(range(len(metric_matrix))) if metric_matrix[i] > thresh]
        to_nodes = []
        count = 0
        for i,k in enumerate(range(len(metric_matrix))):
            if metric_matrix[i] > thresh:
                to_nodes.append(calc_col_idx(k, from_nodes[count], n))
                count += 1
                
        metric_df = pd.DataFrame({'from':np.array(labels)[from_nodes],
                                  'to':np.array(labels)[to_nodes],
                                 metric_name:metric_matrix[metric_matrix > thresh]})
        
        return(metric_df)

    def pairwise_levenshtein(self, entities,entity_labels=None, thresh=None):

        if entity_labels is None:
            entity_labels = entities

        entity_labels = list(entity_labels)

        if thresh is None:
            thresh = 0.0

        transformed_strings = np.array(entities).reshape(-1,1)
        lev_matrix = pdist(transformed_strings, lambda x,y: ratio(x[0],y[0]))
        lev_matrix = squareform(lev_matrix)
            
        lev_df = self.pairwise_metric(lev_matrix, entity_labels, "lev_sim", thresh)
            
        return(lev_df.sort_values('lev_sim'))
    
    def char_ngrams(self, string, n=1):
        string = re.sub(r'[,-./]|\sBD',r'', string)
        ngrams = zip(*[string[i:] for i in range(n)])
        return [''.join(ngram) for ngram in ngrams]
    
    def pairwise_char_tfidf(self, entities,entity_labels=None, thresh=None):
        
        if entity_labels is None:
            entity_labels = entities

        entity_labels = list(entity_labels)

        if thresh is None:
            thresh = 0.0

        vectorizer = TfidfVectorizer(min_df=0.005, analyzer=self.char_ngrams)
        tfidf = vectorizer.fit_transform(entities).todense()
        cosine_similarities = 1.0 - pdist(tfidf,'cosine')
      
        char_tfidf_df = self.pairwise_metric(cosine_similarities, entity_labels, "char_tfdf_cos", thresh)
            
        return(char_tfidf_df.sort_values('char_tfdf_cos'))
    
    def matrix_cosine(self, x, y):
        return np.einsum('ij,ij->i', x, y) / (
                  np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
        )
    
    def tfidf(self, pair_df):
    
        pairs = pair_df[[f'{self.text_field}_clean_from',
                         f'{self.text_field}_clean_to']].copy()
        unique_names = np.unique(pairs.values.flatten())

        tfidf = TfidfVectorizer().fit(unique_names)
        
        CHUNK_SIZE = 50000
        index_chunks = chunked(pairs.index, CHUNK_SIZE)
        chunks = []
        for ii in index_chunks:
            chunk = pairs.iloc[ii]
            tfidf_from = tfidf.transform(chunk[f'{self.text_field}_clean_from']).todense()
            tfidf_to = tfidf.transform(chunk[f'{self.text_field}_clean_to']).todense()
        
            cos_sim_chunk = self.matrix_cosine(tfidf_from, tfidf_to)
            chunks.append(cos_sim_chunk)
            
        cos_sim = np.concatenate(chunks)
        
        return(cos_sim)
    
    def levenshtein(self,pair_df):
        
        lev_sim = pair_df.apply(lambda row: ratio(row[f'{self.text_field}_clean_from'], 
                                                  row[f'{self.text_field}_clean_to']),
                               axis=1)
    
        return(lev_sim)

    def clean_parts(self,x):
    
        x = re.sub(r"[,.;@#?!&$~]+", ' ', x)
        x = re.sub(r"\s+", ' ', x)
    
        x = x.lower()
    
        x = [w for w in re.split('\W+',x) if w != '' and w != ' ']
    
        return(x)
    
    def name_part_graph(self,field):

        name_parts = self.nodes[self.nodes['nodeType'] == self.node_type][field].apply(self.clean_parts)
        
        names = self.nodes[self.nodes['nodeType'] == self.node_type][[field]]
        names['parts']  = name_parts
        names = names.set_index(field)['parts'].explode().reset_index()
        
        name_graph = nx.Graph()

        for part in names['parts'].unique():
            name_graph.add_node(part)

            
        print('Adding edges to name graph...')
        for g,grp in tqdm.tqdm(names.groupby(field)):
            for pair in itertools.combinations(grp['parts'].values, r=2):
        
                if pair[0] != pair[1]:
                    name_graph.add_edge(pair[0],pair[1])


        name_part_likelihoods = names['parts'].value_counts() / names['parts'].value_counts().sum()
        name_part_likelihoods = name_part_likelihoods.reset_index()
        name_part_likelihoods.columns = ['part','likelihood']

        print('Getting name part pair connectivities...')
        connectivities = []
        pairs = []
        name_list = []
        for g,grp in tqdm.tqdm(names.groupby(field)):
    
            for pair in itertools.combinations(grp['parts'].values, r=2):
                name1 = pair[0]
                name2 = pair[1]
                if name1 != name2:
                    conn = approx.local_node_connectivity(name_graph, name1,name2)
                    connectivities.append(conn)
                else:
                    connectivities.append(1)

                pairs.append(f'{name1}-{name2}')
                name_list.append(g)
                    
                    
        name_conns = pd.DataFrame({'name':name_list, 'pair':pairs,'conn':connectivities}).sort_values('conn')
        name_conns['part'] = name_conns['pair'].str.split('-')
        name_conns['conn'] = name_conns['conn'] / name_conns['conn'].max()


        #self.max_conn = name_conns['conn'].max()
        
               
        return(name_part_likelihoods, name_conns)
                    
    
    def name_likelihood(self,field):

        name_conns = self.name_connectivities[f'{field}_clean']

        conn_df = name_conns.copy()

        conn_df = conn_df.explode('part')

        conn_df = conn_df.merge(self.name_part_likelihoods[f'{field}_clean'],
                                on='part')
        
        def get_grp_likelihood(grp):
            means = grp.groupby('part')[['conn','likelihood']].mean()
            means['mean'] = means['conn']*means['likelihood']
            return(means['mean'].mean())

        likelihoods = []
        name_list = []
        for g,grp in tqdm.tqdm(conn_df.groupby('name')):
            l = get_grp_likelihood(grp)
            likelihoods.append(l)
            name_list.append(g)
            
        likelihoods = pd.DataFrame({'name':name_list,'likelihood':likelihoods})

        return(likelihoods)
        
            
    def name_part_sim(self,name1,name2):
    
        parts1 = re.findall(r"[\w']+|[.,!?;]", name1)
        parts2 = re.findall(r"[\w']+|[.,!?;]", name2)

        parts1 = np.array([p for p in parts1 if p not in string.punctuation])
        parts2 = np.array([p for p in parts2 if p not in string.punctuation])

        if len(parts1) > 4 or len(parts2) > 4:
            return(0.0)
        
        if len(parts1) < len(parts2):
            temp = parts1
            parts1 = parts2
            parts2 = temp

        dists = cdist(parts2.reshape(-1, 1), parts1.reshape(-1, 1), lambda x, y: ratio(x[0],y[0]))

        ind1 = list(range(len(parts1)))
        ind2 = list(range(len(parts2)))

        maps = [list(zip(ind2, p)) for p in itertools.permutations(ind1)]

        max_sim = 0.0
        for ind in maps:
            sim = np.mean(dists[tuple(np.array(ind).T.tolist())])
            if sim > max_sim:
                max_sim = sim
                best_map = ind

        return(max_sim)

    def predict_sims(self,name_df):
        col1 = [c for c in name_df.columns if 'from' in c][0]
        col2 = [c for c in name_df.columns if 'from' in c][0]
        name_df['text'] = name_df.apply(lambda x: x[col1] + '+++' + x[col2],axis=1)
        name_df['encoded'] = name_df['text'].apply(lambda x: np.array(self.encode_text(x))[0])
        name_df['length'] = name_df['text'].apply(lambda x: np.array(self.encode_text(x))[1])
        text = torch.from_numpy(np.vstack(name_df['encoded'].values))
        length = torch.from_numpy(name_df['length'].values)

        match_probs = self.model(text,length).squeeze(1).detach().numpy()

        print(match_probs)
        
        return(match_probs)

        
    def get_initials(self,text):
        
        parts = re.split('\W+',text)
        initials = [p[0].lower() for p in parts if len(p) > 0 and str.isalpha(p[0])]
        
        return(initials)

    def char_overlap(self,text1,text2):

        common_letters = Counter(text1) & Counter(text2)
        common = sum(common_letters.values())

        mismatched_letters1 = (Counter(text1) | Counter(text2)) - Counter(text1)
        mismatched_letters1 = sum(mismatched_letters1.values())
        
        mismatched_letters2 = (Counter(text1) | Counter(text2)) - Counter(text2)
        mismatched_letters2 = sum(mismatched_letters2.values())

        score = common - min([mismatched_letters1,mismatched_letters2])
        
        length = min([len(text1),len(text2)])

        score = score/float(length) - 1.0 

        if text1 == text2:
            score += 1.0
        else:
            test1 = [t for t in text1 if t in text2]
            test2 = [t for t in text2 if t in text1]

            if test1 == test2:
                score += 0.5

        score = score - 1.0
                
        return(score)
        
    def initials(self,pair_df):

        pairs = pair_df[[f'{self.text_field}_clean_from',
                         f'{self.text_field}_clean_to']].copy()

        pairs['from_initials'] = pairs[f'{self.text_field}_clean_from'].apply(self.get_initials)
        pairs['to_initials'] = pairs[f'{self.text_field}_clean_to'].apply(self.get_initials)

        tqdm.tqdm.pandas()
        initials_overlap = pairs.progress_apply(lambda row: self.char_overlap(row['from_initials'],row['to_initials']),axis=1)

        return(initials_overlap)
        
        
    def get_text_similarities(self):
        
        idx = self.nodes['nodeType'] == self.node_type
        names = self.nodes[idx][f'{self.text_field}_clean']
        ids = self.nodes[idx][self.id_field]
        
        print('Character-level tfidf...',end="")
        start = time.time()
        self.text_sims = self.pairwise_char_tfidf(names,
                                                  entity_labels=ids,
                                                  thresh=self.candidate_thresh)
        end= time.time()

        self.text_sims = self.text_sims.merge(self.nodes[[self.id_field,
                                         f'{self.text_field}_clean']],
                                  left_on='from',
                                  right_on=self.id_field,
                                  how='left').drop(self.id_field,axis=1).merge(self.nodes[[self.id_field,
                                                                                 f'{self.text_field}_clean']],
                                                      left_on='to',right_on=self.id_field,
                                                      how='left',suffixes=('_from','_to')).drop(self.id_field,axis=1)
        end = time.time()
        print(f'{end-start:.2f} s...',end="")
        print(f'{len(self.text_sims)} pairs')
        
        self.text_sims['text_sim_total'] = 0.0
        for field, metrics in self.steps.items():
            self.text_similarities(field, metrics)

        if not self.blacklist is None:
            self.text_sims = self.text_sims.merge(self.blacklist,
                                                  how='left').fillna(False)
        else:
            self.text_sims['blacklist'] = False
        
        return(self.text_sims)

    def filter_sims(self,field,thresh):

        self.text_sims = self.text_sims[self.text_sims[field] > thresh]
        
    def text_similarities(self, field, metrics):
        
        if 'tfidf' in metrics:
            print('Word-level tfidf...', end="")
            start = time.time()
            self.text_sims[f'{field}_tfidf_sim'] = self.tfidf(self.text_sims)
            self.text_sims[f'{field}_tfidf_sim'] =  self.text_sims[f'{field}_tfidf_sim'].fillna(0.0)
            end = time.time()
            print(f'{end-start:.2f} s')
            self.text_sims['text_sim_total'] += self.text_sims[f'{field}_tfidf_sim']
        if 'lev' in metrics:
            print('Levenshtein...', end="")
            start = time.time()
            self.text_sims[f'{field}_lev_sim'] = self.levenshtein(self.text_sims)
            self.text_sims[f'{field}_lev_sim'] = self.text_sims[f'{field}_lev_sim'].fillna(0.0)
            end = time.time()
            print(f'{end-start:.2f} s')
            self.text_sims['text_sim_total'] += self.text_sims[f'{field}_lev_sim']
        if 'parts' in metrics:
            print('Name parts similarity...', end="")
            start = time.time()
            tqdm.tqdm.pandas()
            self.text_sims[f'{field}_parts_sim'] = self.text_sims.progress_apply(lambda row: self.name_part_sim(row[f'{self.text_field}_clean_from'], 
                                                                                        row[f'{self.text_field}_clean_to']),
                                                        axis=1)
            self.text_sims[f'{field}_parts_sim'] = self.text_sims[f'{field}_parts_sim'].fillna(0.0)
            end = time.time()
            print(f'{end-start:.2f} s')
            self.text_sims['text_sim_total'] += self.text_sims[f'{field}_parts_sim']
        if 'predict' in metrics and self.run_predict:
            print('Predicted match probability...',end="")
            start = time.time()
            self.text_sims[f'{field}_predict_sim'] = self.predict_sims(self.text_sims[[f'{self.text_field}_clean_from',
                                                                                        f'{self.text_field}_clean_to']]) 
            self.text_sims[f'{field}_predict_sim'] = self.text_sims[f'{field}_predict_sim'].fillna(0.0)

            end= time.time()
        if 'initials' in metrics:
            print('Initials...')
            start = time.time()
            self.text_sims[f'{field}_initials_sim'] = self.initials(self.text_sims)
            self.text_sims[f'{field}_initials_sim_percentile'] = self.text_sims[f'{field}_initials_sim'].rank(pct=True,
                                                                                                              method='max')
            end = time.time()
            print(f'{end-start:.2f} s')
            self.text_sims['text_sim_total'] *= self.text_sims[f'{field}_initials_sim_percentile']

        if 'likelihood' in metrics:
            print('Name likelihoods...',end="")
            start = time.time()

            unique_names = pd.unique(self.text_sims[[f'{self.text_field}_clean_from',
                                                    f'{self.text_field}_clean_to']].values.ravel('K'))
            unique_names = pd.DataFrame({'name':unique_names})
            
            likelihoods = self.name_likelihood(self.text_field)
            unique_names = unique_names.merge(likelihoods,on='name')
            
            from_likelihood = self.text_sims.merge(unique_names,left_on=f'{self.text_field}_clean_from',
                                                   right_on='name',how='left')['likelihood']
            to_likelihood = self.text_sims.merge(unique_names,left_on=f'{self.text_field}_clean_to',
                                                   right_on='name',how='left')['likelihood']

            self.text_sims[f'{field}_likelihood'] = (from_likelihood + to_likelihood) / 2.
#            self.text_sims[f'{field}_likelihood'] = 1.0 - (1.0 - (self.text_sims[f'{field}_likelihood'] /  self.text_sims[f'{field}_likelihood'].max())).pow(4)
            self.text_sims[f'{field}_likelihood'] = (1.0 -  self.text_sims[f'{field}_likelihood'] /  self.text_sims[f'{field}_likelihood'].max()).pow(4)
            end=time.time()
            print(f'{end-start:.2f} s')
            self.text_sims['text_sim_total'] *= self.text_sims[f'{field}_likelihood']
            
        print(fr'{len(self.text_sims)} pairs')
    
    def get_sim_communities(self,metric,thresh):
        
        merged_graph=nx.from_pandas_edgelist(self.text_sims[ (self.text_sims[metric] > thresh) & (~self.text_sims['blacklist'])], 
                                             'from', 'to', metric)

        self.communities = sorted(nxcom.greedy_modularity_communities(merged_graph,weight=metric), 
                                 key=len, reverse=True)
        
    def show_communities(self):
        
        self.display_comms = []
        print(f'{len(self.communities):,}' + ' communities')
        res = pd.DataFrame(columns=['clusterID','nEdges','nNodes','nodeDisplayNames'])
        for c,comm in enumerate(self.communities):
            display_comm = []
            nEdgesImpacted = 0 
            for node in comm:
                dc = self.nodes[self.nodes[self.id_field] == node][self.text_field].values[0]
                display_comm.append(dc)
                nEdgesImpacted = nEdgesImpacted + len(self.edges[(self.edges['from']==node)|
                                                                 (self.edges['to']==node)])
            res.loc[len(res)] = [c, nEdgesImpacted, len(display_comm), display_comm]
            #print(c,display_comm) 
            self.display_comms.append(display_comm)
        for c in ['nEdges','nNodes']:
            res[c] = res[c].apply(lambda x:f'{x:,}')
        display(res.sort_values(by='nEdges', ascending=False))
        return(self.display_comms)


    def add_to_blacklist(self,node1,node2, fn = None):

        if self.blacklist_fn is None and fn is None:
            fn = input("Enter file name to store blacklist: ")
            self.blacklist_fn = fn
        elif fn is None:
            fn = self.blacklist_fn
        
        node1 = self.nodes[self.nodes['displayName'] == node1]['nodeID'].values[0]
        node2 = self.nodes[self.nodes['displayName'] == node2]['nodeID'].values[0]

        blacklist = pd.DataFrame({'from':[node1,node2],
                                  'to':[node2,node1],
                                  'blacklist':[True,True]})

        self.blacklist = pd.concat([self.blacklist, blacklist]).drop_duplicates()


        self.text_sims.loc[ (self.text_sims['from'] == node1) & (self.text_sims['to'] == node2), 'blacklist'] = True
        self.text_sims.loc[ (self.text_sims['from'] == node2) & (self.text_sims['to'] == node1), 'blacklist'] = True
                                

        self.blacklist.to_csv(fn,index=False)
        

    def add_cluster_to_blacklist(self,cluster_num,fn=None):

        if self.blacklist_fn is None and fn is None:
            fn = input("Enter file name to store blacklist: ")
            self.blacklist_fn = fn
        elif fn is None:
            fn = self.blacklist_fn
        
        comm = list(self.communities[cluster_num])

        blacklist = pd.DataFrame(itertools.product(comm,comm))
        blacklist.columns = ['from','to']
        blacklist['blacklist'] = True
        blacklist = blacklist[blacklist['from'] != blacklist['to']]
        
        self.blacklist = pd.concat([self.blacklist, blacklist]).drop_duplicates()

        idx = self.text_sims.set_index(['from','to']).index.isin(self.blacklist.set_index(['from','to']).index)
        self.text_sims.loc[idx,'blacklist'] = True
                                        

        self.blacklist.to_csv(fn,index=False)


    def partial_cluster_blacklist(self,cluster_num,keeps = [], fn=None):

        if self.blacklist_fn is None and fn is None:
            fn = input("Enter file name to store blacklist: ")
            self.blacklist_fn = fn
        elif fn is None:
            fn = self.blacklist_fn
        
        comm = list(self.communities[cluster_num])


        all_keep_ids = []
        for keep in keeps:

            keep_ids = [self.nodes.loc[self.nodes['displayName'] == k,'nodeID'].values[0] for k in keep]
            all_keep_ids += keep_ids
            
            not_keep = [c for c in comm if c not in keep_ids]
            
            blacklist = pd.DataFrame(itertools.product(keep_ids,not_keep))
            blacklist.columns = ['from','to']
            blacklist['blacklist'] = True
            temp = blacklist.copy()
            temp.columns = ['to','from','blacklist']
            
            self.blacklist = pd.concat([self.blacklist, blacklist, temp]).drop_duplicates()


        not_keeps = [c for c in comm if c not in all_keep_ids]
        blacklist = pd.DataFrame(itertools.product(not_keep,not_keep))
        blacklist.columns = ['from','to']
        blacklist['blacklist'] = True
        
        self.blacklist = pd.concat([self.blacklist, blacklist]).drop_duplicates()

        idx = self.text_sims.set_index(['from','to']).index.isin(self.blacklist.set_index(['from','to']).index)
        self.text_sims.loc[idx,'blacklist'] = True                                

        self.blacklist.to_csv(fn,index=False)

        

        
        
    def merge_nodes(self):
        
        print(f'Before merging there are {len(self.nodes)} nodes')
        
        try:
            self.communities
        except:
            self.get_sim_communities()
        
        nodeIDsreplaced={}
        for comm in self.communities:
            
            if len(comm) < 2:
                continue

            nodes_subset = self.nodes[self.nodes['nodeID'].isin(comm)]
            
            from_subset = self.edges[self.edges['from'].isin(comm)]
            to_subset = self.edges[self.edges['to'].isin(comm)]
            
            from_counts = from_subset['from'].value_counts().reset_index().rename(columns = {"from":'count'})
            to_counts = to_subset['to'].value_counts().reset_index().rename(columns = {"to":'count'})
            
            counts = pd.concat([from_counts,to_counts]).groupby('index')['count'].sum().reset_index()

            all_nodes = pd.DataFrame({'index':list(comm)})

            counts = counts.merge(all_nodes,on='index',how='outer').fillna(0.0).sort_values('count')
            
            if len(counts) > 0:                
                canonical = counts.tail(1)['index'].values[0]
                replace = counts.head(-1)['index'].values
            else:                
                canonical = list(comm)[0]
                replace = list(comm)[1:]

            if not self.merge_log is None:
                pairs = pd.DataFrame(list(itertools.combinations(comm, 2)))
                pairs.columns = ['match1','match2']
                pairs['canonical'] = canonical
                self.merge_log = pd.concat([self.merge_log,pairs]).reset_index(drop=True)
                self.merge_log.to_csv(self.merge_fn,index=False)

                
            nodeIDsreplaced[canonical] = replace

            for node in replace:

                self.edges['from'] = self.edges['from'].replace(node,canonical)
                self.edges['to'] = self.edges['to'].replace(node,canonical)
                
                self.nodes = self.nodes[self.nodes[self.id_field] != node]
                
        self.edges = self.edges.drop_duplicates()
        print(f'After merging there are {len(self.nodes)} nodes')
        return nodeIDsreplaced

    def save_context_graph(self, node_file, edge_file):

        nodes = [json.dumps(row.dropna().to_dict()) for index,row in self.nodes.iterrows()]
        with open(node_file,'w') as f:
            for node in nodes:
                f.write(node + '\n')

        edges = [json.dumps(row.dropna().to_dict()) for index,row in self.edges.iterrows()]
        with open(edge_file,'w') as f:
            for edge in edges:
                f.write(edge + '\n')

            