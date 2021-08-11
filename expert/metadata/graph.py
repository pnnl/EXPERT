"""

Direct Links:

'Published In', 'Related To', 'Funded By', 'Author of', 'Attended', 'Located At',  'Belongs To', 'Expert In'

Indirect links: 

'Co-author', 'Co-attendee', 'Colleague'


"""

import os
import graphviz
from graphviz import Digraph, Graph
import pandas as pd 
from IPython.core.display import display, HTML
from graphviz import Source  


def weight_dataframe(dataframe, source_node_column_name, target_node_column_name, edge_label_column_name, 
                     weight_col='weight', weighted=True):
    dataframe = dataframe.copy()
    if weight_col not in dataframe.columns:
        dataframe[weight_col] = 1
    dataframe = dataframe.groupby([source_node_column_name,
                              target_node_column_name,
                              edge_label_column_name],as_index=False)[weight_col].sum()
    if weighted: 
        return dataframe
    else:
        dataframe[weight_col] = 1
        return dataframe

directEdgeTypes =  ['Author of', 'Attended', 'Belongs To', 'Expert In', 
                    'Funded By', 'Located At', 'Published In', 'Related To']
        
nodeTypeColors = {
    'Paper':'gray',
    'Scientist':'skyblue', 
    'Topic/Expertise':'yellow', 
    'Journal':'orange', 
    'Conference':'tomato', 
    'Location':'violet', 
    'Institution':'aquamarine',
    'Funding Agency':'darkseagreen',
    'Book':'lavender'}

nodeTypeShapes = {
    'Paper':'note', 
    'Scientist':'egg', 
    'Topic/Expertise':'oval', 
    'Journal':'oval', 
    'Book':'oval', 
    'Conference':'oval', 
    'Location':'box', 
    'Institution':'oval',
    'Funding Agency':'oval'}

edgeTypeColors = {
    'Author Of':'black',#'skyblue', 
    'Attended':'black',#'tomato', 
    'Belongs To':'black',#'blue',
    'Co-author':'black',#'cyan', 
    'Co-attendee':'black',#'mistyrose',
    'Colleague':'black',#'violet',
    'Expert In':'black',#'gold',
    'Funded':'black',#'green', 
    'Located At':'black',#'indigo', 
    'Published In':'black',#'lightcoral',
    'Related To':'black',#'orangered'
                 }


class metadataGraph:
    """
    MetaData Graph Generation and Visualization
    """    
    def __init__(self, links, nodes, 
                 engine='sfdp', weighted=False, TB=True, figureSize=10, nodeSize=.1, scaleEdges=1,
                 name='metadata graph', directed=True,
                 nodeTypeColors=nodeTypeColors, nodeTypeShapes=nodeTypeShapes, edgeTypeColors=edgeTypeColors,
                 displayLabels=True, safeNodeLabels=True, truncateLabels=-1):\
        
        self.sourceNodeCol = 'from'
        self.targetNodeCol = 'to'
        self.edgeTypeCol = 'edgeType'
        
        if type(links) == str:
            if not os.path.exists(links): print(f'Links file does not exist: {links}')
            file_type = links.split('.')[-1]
            if file_type in ['json','jsonl']:  
                links = pd.read_json(links, lines=True)
            if file_type in ['csv']:   
                links = pd.read_csv(links)
        self.links = links
        
        self.weight_col='edgeWeight'
        self.links = weight_dataframe(self.links, self.sourceNodeCol, self.targetNodeCol, self.edgeTypeCol, 
                                      weight_col=self.weight_col, weighted=weighted)
        
        nodeIDs = list(set(self.links[self.sourceNodeCol]).union(set(self.links[self.targetNodeCol])))
        
        self.scaleEdges = scaleEdges
                     
        if type(nodes) == str:
            if not os.path.exists(nodes): print(f'Nodes file does not exist: {nodes}')
            file_type = nodes.split('.')[-1]
            if file_type in ['json','jsonl']:  
                nodes = pd.read_json(nodes, lines=True)
            if file_type in ['csv']:   
                nodes = pd.read_csv(nodes)
                        
        self.nodes = nodes[nodes['nodeID'].isin(nodeIDs)].copy()
        self.nodeDisplayNames = self._nodeDisplayNames(displayLabels=displayLabels) 
        
        self.default_engine = engine
        self.TB = TB
        self.size = figureSize
        self.name = name 
        self.nodeSize = str(nodeSize)
        
        self.nodeTypeColors=nodeTypeColors
        self.nodeTypeShapes=nodeTypeShapes
        self.edgeTypeColors=edgeTypeColors
        self.displayLabels = displayLabels
        
        self.safeNodeLabels = safeNodeLabels
        self.truncateLabels = truncateLabels
        self.directed = directed
        
        self.graphObj = self._graph()
        
    def _formatDisplayLabel(self, label):
        if self.safeNodeLabels:
            label = str(label).replace(':',' - ')
        if not self.displayLabels:
            #label = ' '
            label = ''
        if self.truncateLabels:
            if self.truncateLabels > -1 and len(label) > self.truncateLabels:
                label = label[:self.truncateLabels]+'..'
        return label 
    
    def _nodeDisplayNames(self,displayLabels=True):
        nodeDisplayNames = {}
        for nodeType, nodesDf in self.nodes.groupby('nodeType'):
            nodeDisplayNames[nodeType] = {nodeID:nodeDisplayName for nodeID, nodeDisplayName in 
                                          zip(nodesDf['nodeID'], nodesDf['displayName'])}
        return nodeDisplayNames
          
    def _graphAttributes(self):
        if self.TB:
            graph_attr={'rankdir':'TB', 'size':f'{self.size},{self.size}!'}
            #graph_attr={'rankdir':'TB', 'size':str(self.size)}
        else:
            graph_attr={'rankdir':'LR', 'size':f'{self.size},{self.size}!'}
            #graph_attr={'rankdir':'LR', 'size':str(self.size)}
        return graph_attr  
         
    def _graph(self):  
        if self.directed:
            # instantiate directed graph
            g = Digraph(name=self.name, engine=self.default_engine, graph_attr=self._graphAttributes())
        else:
            g = Graph(name=self.name, engine=self.default_engine, graph_attr=self._graphAttributes())
        
        self.sourceNodeCol = 'from'
        self.targetNodeCol = 'to'
        self.edgeTypeCol = 'edgeType'
                
        # add nodes, color and shape by type and label by node displayName specified in nodes data
        for nodeType in self.nodeDisplayNames:
            nodeColor = self.nodeTypeColors[nodeType]
            nodeShape = self.nodeTypeShapes[nodeType]
            nodeMapping = self.nodeDisplayNames[nodeType] 

            for nodeID in nodeMapping:
                nodeLabel = self._formatDisplayLabel(str(nodeMapping[nodeID]))
                g.node(str(nodeID), label=nodeLabel, shape=nodeShape, color=nodeColor,
                       style='filled', width=self.nodeSize, height=self.nodeSize,
                       fixedsize=str(not self.displayLabels).lower())
        
        # add edges, color and label by type 
        self.links = self.links.reset_index()
        for i in self.links.index:
            edgeType = str(self.links.loc[i, self.edgeTypeCol])
            weight = str(self.links.loc[i, self.weight_col] * self.scaleEdges)
            src = str(self.links.loc[i,self.sourceNodeCol])
            dst = str(self.links.loc[i, self.targetNodeCol])   
            edgeColor = self.edgeTypeColors[edgeType] 
            
            g.edge(src, dst, 
                   label=self._formatDisplayLabel(edgeType), penwidth = str(weight),
                   arrowsize=str(self.scaleEdges), color = edgeColor) 
            
        return g

    def graph(self):
        return self.graphObj
         
    def _set_graph_parameters(self, graph, engine=None, TB=None, figureSize=None):
        # specify engine
        if engine is not None and graph.engine != engine:
            graph.engine = engine 
        # set Left-to-Right or Top-to-Bottom
        if TB is not None:
            if TB:
                graph.graph_attr['rankdir']='TB' 
            else:
                graph.graph_attr['rankdir']='LR' 
                
        if figureSize is not None:
            graph.graph_attr['size']=f'{figureSize},{figureSize}!' 
            print(graph.graph_attr['size'])
            #            graph.graph_attr['size']=str(figureSize)

        return graph
        
    def _get_graphviz(self, engine=None, TB=None, figureSize=None):
        # generate graphviz graph using the specified engine
        graphviz_graph = self.graph().copy()
        # set parameters
        graphviz_graph = self._set_graph_parameters(graphviz_graph, engine=engine, TB=TB, figureSize=figureSize)
        return graphviz_graph
       
        
    def legend(self, figureSize=None, show=True, returnGraphviz=False,
               fn='legend', savedir='./', pdf=False, png=False, dot=False):  
        # instantiate directed graph
        engine = 'neato'
        l = Digraph(name=self.name+' legend', 
                    engine=engine, 
                    graph_attr=self._graphAttributes())
        
        # set parameters
        l = self._set_graph_parameters(l, engine=engine, TB=None, figureSize=figureSize)
        
        for nodeType in self.nodeDisplayNames:
            nodeColor = self.nodeTypeColors[nodeType]
            nodeShape = self.nodeTypeShapes[nodeType]
            nodeMapping = self.nodeDisplayNames[nodeType] 
            
            l.node(str(nodeType), 
                   label = nodeType,
                   #label=' ', pos="0,0", xlabel=nodeType, 
                   shape=nodeShape, color=nodeColor, style='filled', width=self.nodeSize, height=self.nodeSize)
        if show: 
            display(l)
        if pdf or png or dot:
            # write to pdf
            filename=os.path.join(savedir, fn)
            if pdf: 
                l.format = 'pdf'
                l.render(filename=filename)  
            if png:
                l.format = 'png'
                l.render(filename=filename)   
            # remove dot file unless specified to keep it
            if not dot: os.remove(filename)
            
        if returnGraphviz: return l
                
    def visualize(self, engine=None, TB=None, figureSize=None):
        """
        Visualize metadata graph  
        """
        # generate graphviz graph using the specified engine and args
        graphviz_graph = self._get_graphviz(engine=engine, TB=TB, figureSize=figureSize)
        # display 
        display(graphviz_graph)
    
    
    def save(self, engine=None, TB=None, figureSize=None, 
             fn='graph', savedir='./', pdf=True, png=False, dot=False):
        """
        Save visualization of metadata graph using specific Graphviz engine (default is 'sfdp') to PDF,
        using specified filename (fn, default is 'graph.pdf')
        """ 
        # check that save_dir directory exists, otherwise create it
        os.makedirs(savedir, exist_ok=True)
        
        # generate graphviz graph using the specified engine and args
        graphviz_graph = self._get_graphviz(engine=engine, TB=TB, figureSize=figureSize)
        
        # write to pdf
        filename=os.path.join(savedir, fn)
        if pdf: 
            graphviz_graph.format = 'pdf'
            graphviz_graph.render(filename=filename)  
        if png:
            graphviz_graph.format = 'png'
            graphviz_graph.render(filename=filename)   
        # remove dot file unless specified to keep it
        if not dot: os.remove(filename)
     
         
 

