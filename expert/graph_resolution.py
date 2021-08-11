import pandas as pd
import networkx as nx
import networkx.algorithms.community as nxcom
from vis import *
from sklearn.metrics import pairwise


def node_similarity_graph(nodes_with_embeddings, threshold=40, col='VERB',plot=False):
    
    pairwise_dist = pairwise.cosine_similarity(nodes_with_embeddings['value'].tolist())
    adj = pairwise_dist * (pairwise_dist > threshold)
    graph = nx.from_numpy_matrix(np.matrix(adj)) 
    
    communities = sorted(nxcom.greedy_modularity_communities(graph,weight='weight'), 
                         key=len, reverse=True)
    
    if plot:
        
        dataframe = nx.convert_matrix.to_pandas_edgelist(graph)
        dataframe = dataframe[dataframe['source'] != dataframe['target']]
        
        g = draw_network(dataframe,nodes_with_embeddings,communities)
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
        subset = full_df[full_df[col].isin(choices)]
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




def main():

    df = pd.read_csv('file_with_embeddings.csv')

    temp = df.drop_duplicates(subset=['ARG'])
    comms, sims = node_similarity_graph(temp,threshold=0.87,col='ARG')

    for sim,comm in [(s,x) for s, x in sorted(zip(sims,comms), key=lambda pair: pair[0])]:
        if len(comm) > 1:
            print(sim,temp.iloc[list(comm)]['ARG'].unique())

    df2 = replace_values(comms,sims,temp,full_df=df,col='ARG',thresh=0.87)
    
    df_replace = df.merge(df[['doc_id','sentence_id','ARG0','ARG1','VERB','variable','ARG']],
                          on=['doc_id','sentence_id','ARG0','ARG1','VERB','variable'],
                          suffixes=('','_new'))


if __name__ == "__main__":
    main()
