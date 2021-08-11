import pandas as pd
import networkx as nx
import networkx.algorithms.community as nxcom
from .vis import *
from sklearn.metrics import pairwise
import numpy as np
from graphviz import Graph

from matplotlib import pyplot as plt
import seaborn as sns


def draw_network(dataframe,df,communities, col='ARG'):
    source_node_column_name = 'source'
    target_node_column_name = 'target'
    weight_col = 'weight'
    dataframe['weight'] = dataframe['weight'] - 0.6
    g = Graph('G', engine='sfdp')
    colors = sns.color_palette()
    for i in dataframe.index:

        source = dataframe.loc[i,source_node_column_name]
        target = dataframe.loc[i,target_node_column_name]
        g.edge(df.loc[source,col],
               df.loc[target,col],
               #label=dataframe.loc[i, edge_label_column_name],
               penwidth = str(dataframe.loc[i, weight_col]*5.0))
        
    for i, comm in enumerate(communities):
        for node in comm:
            g.node(df.loc[node,col],color=str(i + 1),
                   colorscheme="dark28",penwidth='3')
                                                                                                                                
    return g


def node_similarity_graph(nodes_with_embeddings, threshold=40, col='VERB',plot=False):
    
    pairwise_dist = pairwise.cosine_similarity(nodes_with_embeddings['value'].tolist())
    adj = pairwise_dist * (pairwise_dist > threshold)
    graph = nx.from_numpy_matrix(np.matrix(adj)) 
    
    communities = sorted(nxcom.greedy_modularity_communities(graph,weight='weight'), 
                         key=len, reverse=True)
    
    if plot:
        
        dataframe = nx.convert_matrix.to_pandas_edgelist(graph)
        dataframe = dataframe[dataframe['source'] != dataframe['target']]
        
        g = draw_network(dataframe,nodes_with_embeddings,communities,col=col)
        visualize(g)
        save_to_pdf(g, 'similarity_graph_example', save_dir='./')
    
    sim_scores = []
    for comm in communities:
        comm_sims = pairwise_dist[np.array(list(comm)),:]
        comm_sims = comm_sims[:,np.array(list(comm))]
        sim_scores.append(np.mean(comm_sims))
        
    return(communities,sim_scores)
    

def get_replace_dict(comms,df,col='ARG'):
    
    replace_dict = {}
    for comm in comms:
        choices = list(df.iloc[list(comm)][col].unique())
        if len(choices) > 1:
            subset = df.iloc[list(comm)]
            subset = subset[col].value_counts().reset_index()
            subset = subset[subset[col] == subset[col].max()]
            subset['len'] = subset['index'].apply(lambda x: len(x))
            subset = subset.sort_values('len')
            selection = subset.head(1)['index'].values[0]
            for choice in choices:
                if choice != selection:
                    replace_dict[choice] = selection

    return(replace_dict)

def replace_values(comms,sims,df,full_df=None,col='ARG',thresh=0.8):
    
    subsets = []
    for i,comm in enumerate(comms):
         
        choices = list(df.iloc[list(comm)][col].unique())
        subset = full_df[full_df[col].isin(choices)].copy()
        if sims[i] > thresh:
            counts = subset[col].value_counts().reset_index()
            print(counts)
            counts = counts[counts[col] == counts[col].max()]
            counts['len'] = counts['index'].apply(lambda x: len(x))
            counts = counts.sort_values('len')
            print(counts)
            selection = counts.head(1)['index'].values[0]
            print('----'*10)
            for choice in choices:
                if choice != selection:
                    subset[col] = subset[col].replace({choice:selection})
        subsets.append(subset)

    return(pd.concat(subsets))


def explore_weights(graph_df,limit=5.0):
    graph_df['weight'].hist(grid=False)
    plt.xlabel('Edge Weight')
    plt.ylabel('Number of Edges')
    plt.savefig('edge_weight_hist_example.png',bbox_inches="tight")
    plt.show()
    
    sorted_df = graph_df.sort_values('weight').reset_index(drop=True)
    
    cumprop = sorted_df['weight'].cumsum() / graph_df['weight'].sum()
    plt.plot(sorted_df['weight'],1.0 - cumprop)
    plt.xlabel('Edge Weight Threshold')
    plt.ylabel('Proportion of\nTotal Edge Weight')
    plt.show()
    
    cumprop.name = 'cum_weight_prop' 
    
    merged = pd.concat([sorted_df,pd.DataFrame(cumprop)],
                      axis=1)

        
    merged = merged.drop_duplicates(subset=["weight"],keep='last')

    merged['cum_weight_prop_diff'] = merged['cum_weight_prop'].diff()
    
    
    plt.plot(merged['weight'],1.0 - merged['cum_weight_prop'])
    plt.xlabel('Edge Weight Threshold')
    plt.ylabel('Proportion of\nTotal Edge Weight')
    plt.savefig('cumulative_edge_weight_example.png',bbox_inches="tight")
    plt.show()
    
    plt.plot(merged['weight'],merged['cum_weight_prop_diff'])
    plt.xlabel('Edge Weight Threshold')
    plt.ylabel('Change in Proportion of\nTotal Edge Weight')
    plt.savefig('delta_cumulative_edge_weight_example.png',bbox_inches="tight")
    plt.show()
    
    merged = merged.sort_values("cum_weight_prop_diff",ascending=False)
    merged['rel_cum_weight_prop_diff'] = merged['cum_weight_prop_diff'] / merged['cum_weight_prop_diff'].median()
    
    idx = merged['rel_cum_weight_prop_diff'] > limit
    print(merged[idx][['weight','cum_weight_prop','cum_weight_prop_diff','rel_cum_weight_prop_diff']])
    
    if (merged['rel_cum_weight_prop_diff'] > limit).sum() > 0:
        merged = merged[merged['rel_cum_weight_prop_diff'] > limit]
    else:
        merged = merged.head(1)
    
    selection = merged['weight'].max()
    
    return(selection)


