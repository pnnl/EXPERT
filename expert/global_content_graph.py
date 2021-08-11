import os
import numpy as np

from matplotlib import pyplot as plt

import graphviz
from graphviz import Digraph, Graph
import pandas as pd 
from IPython.core.display import display, HTML
from graphviz import Source  


class globalContentGraph:
    """
    Global Content Graph Generation and Visualization
    """    
    def __init__(self, links, nodes,
                 contentNodeTypes = ['Tag','Keyword','Topic'],
                 keywordFN = None):
        
        self.sourceNodeCol = 'from'
        self.targetNodeCol = 'to'
        self.edgeTypeCol = 'edgeType'

        self.contentNodes = contentNodeTypes
        
        if type(links) == str:
            if not os.path.exists(links): print(f'Links file does not exist: {links}')
            file_type = links.split('.')[-1]
            if file_type in ['json','jsonl']:  
                links = pd.read_json(links, lines=True)
            if file_type in ['csv']:   
                links = pd.read_csv(links)
        self.links = links
                
        nodeIDs = list(set(self.links[self.sourceNodeCol]).union(set(self.links[self.targetNodeCol])))
        
                  
        if type(nodes) == str:
            if not os.path.exists(nodes): print(f'Nodes file does not exist: {nodes}')
            file_type = nodes.split('.')[-1]
            if file_type in ['json','jsonl']:  
                nodes = pd.read_json(nodes, lines=True)
            if file_type in ['csv']:   
                nodes = pd.read_csv(nodes)
                        
        self.nodes = nodes[nodes['nodeID'].isin(nodeIDs)].copy()        
        self.edges = links

        nodeTypes = self.nodes[self.nodes['nodeType'].isin(self.contentNodes)]['nodeType'].unique()
        if len(nodeTypes) > 1:
            colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628',
                      '#f781bf','#999999']
            self.colors = {nt:colors[i] for i, nt in enumerate(nodeTypes)}
        else:
            self.colors = {nodeTypes[0]:'k'}
        
        self.contentToPaper = self._project_graph()
        self.coocc = self._get_cooccur()
        
        
    def _project_graph(self):

        print('Getting content to paper relationships...')
        selectedNodes = self.nodes[self.nodes['nodeType'].isin(self.contentNodes)]
        selectedNodes = list(selectedNodes['nodeID'].values)

        idx = self.edges[self.sourceNodeCol].isin(selectedNodes) | self.edges[self.targetNodeCol].isin(selectedNodes)
        selectedEdges = self.edges[idx]

        selectedEdges = selectedEdges[[self.sourceNodeCol,self.targetNodeCol]]
        selectedEdges['relatedTo'] = True
        
        contentToPaper = pd.pivot_table(selectedEdges,index=self.sourceNodeCol,
                                        columns=self.targetNodeCol, values='relatedTo').fillna(False)

        return(contentToPaper)

    def _get_cooccur(self):
        

        print('Running coocurrence correlation...')
        coocc = ( self.contentToPaper.astype(float) ).corr()
        coocc.columns.name = None
        
        return(coocc)


    def get_edges(self):


        idx = np.triu(np.ones(self.coocc.shape),1).astype(np.bool)
        coocc = self.coocc.where(idx)
        
        coocc = coocc.reset_index()
        
        content_edges = pd.melt(coocc,
                                id_vars=[self.targetNodeCol]).dropna()
        content_edges.columns = ['from','to','weight']
        content_edges = content_edges[content_edges['weight'] > 0.0].sort_values('weight',ascending=False)
        
        self.content_edges = content_edges
        
        return(content_edges)


    def explore_threshold(self,log=False):

        self.content_edges = self.content_edges.sort_values('weight',ascending=False)
        self.content_edges['cml_count'] = 1
        self.content_edges['cml_count'] = self.content_edges['cml_count'].cumsum()

        self.content_edges.plot(x='weight',y='cml_count')
        plt.xlabel('Weight Threshold')
        plt.ylabel('Number of Edges')
        if log:
            plt.yscale("log")

        self.content_edges = self.content_edges.drop('cml_count',axis=1)

    def apply_threshold(self,thresh):

        self.content_edges = self.content_edges[self.content_edges['weight'] >= thresh]

    def save_graph(self, fn):

        content_edges = self.content_edges.copy()
        content_edges = content_edges.merge(self.nodes[['nodeID','nodeType']],
                                            left_on='from',
                                            right_on='nodeID',
                                            how='left')
        content_edges = content_edges.drop('nodeID',axis=1)
        content_edges = content_edges.rename(columns={'nodeType':'fromType'})
        
        content_edges = content_edges.merge(self.nodes[['nodeID','nodeType']],
                                            left_on='to',
                                            right_on='nodeID',
                                            how='left')
        content_edges = content_edges.drop('nodeID',axis=1)
        content_edges = content_edges.rename(columns={'nodeType':'toType'})

        content_edges.to_json(fn,orient='records',lines=True)
        
    def generate_graph(self,engine='sfdp', weighted=True, scale_vis_edge_weight=1.0, 
                       max_edge_weight=None, 
                       filt = 0.0, orient='LR'):
        """

        :param engine: (string) specifies the engine to use for graphviz,
        default is 'sfdp' but can also use 'neato','dot','circo'

        :param weighted: (boolean) flag that indicates whether to collapse edges and represent frequency with
        edge thickness
        :param scale_vis_edge_weight: (float) scale factor to apply to edge weight thickness,
         by default = 1.0 to use raw frequency as thickness

        :return: graphviz Graph
        """

        dataframe = self.content_edges.copy()

        if weighted:
            weight_col = 'weight'
            dataframe = dataframe.groupby([self.sourceNodeCol,
                                      self.targetNodeCol],as_index=False)['weight'].sum()
            dataframe = dataframe[dataframe['weight'] > filt]

        g = Graph('G', engine=engine, graph_attr={'rank':orient})

        idx = (self.nodes['nodeID'].isin(dataframe['to'].values)) | (self.nodes['nodeID'].isin(dataframe['from'].values))
        nodes = self.nodes[idx].copy()
        nodes['color'] = nodes['nodeType'].map(self.colors)
        
        for i in nodes.index:
            g.node(nodes.loc[i,'nodeID'],
                   color=nodes.loc[i,'color'],
                   fontcolor=nodes.loc[i,'color'])

        for i in dataframe.index:
            if weighted:
                if max_edge_weight is not None and dataframe.loc[i, weight_col] > max_edge_weight:

                    g.edge(dataframe.loc[i,self.sourceNodeCol],
                           dataframe.loc[i, self.targetNodeCol],
                           penwidth = str(max_edge_weight * scale_vis_edge_weight)) 
                else:
                    g.edge(dataframe.loc[i,self.sourceNodeCol],
                           dataframe.loc[i, self.targetNodeCol],
                           penwidth = str(dataframe.loc[i, weight_col] * scale_vis_edge_weight))
                #
            else:
                g.edge(dataframe.loc[i,self.sourceNodeCol],
                       dataframe.loc[i, self.targetNodeCol],
                      )
        return g



    def visualize(self,graphviz_graph):
        """
        Save graph visualization to pdf
        :param graphviz_graph: graphviz object for the casual graph
        """
        # display graphviz visualization of causal graph
        src_of_G = graphviz.Source(graphviz_graph)
        display(src_of_G)

    def save_to_pdf(self,graphviz_graph, graph_fname, *, save_dir='./'):
        """
        Save graph visualization to pdf
        :param graphviz_graph: graphviz object for the casual graph
        :param save_dir: directory to save pdf image and gml file for causal graph to
        :param graph_fname: filename to use when saving to pdf e.g "narrative_graph" will
        save the graph to "narrative_graph.pdf"
        """

        # display graphviz visualization of causal graph
        src_of_G = graphviz.Source(graphviz_graph)
        
        # check that save_dir directory exists, otherwise create it
        os.makedirs(save_dir, exist_ok=True)

        # write to pdf
        filename=os.path.join(save_dir, f"{graph_fname}")
        src_of_G.render(filename=filename)
