from .entityMerger import EntityMerger 
from .dedup import deduplication, removeDisplayNamesDefault

from functools import lru_cache 
import pandas as pd 
import networkx as nx
import numpy as np
from expertwidgets import IdentityMatcher
import datetime, math, os, json 
from IPython.display import display, HTML, clear_output

import matplotlib.pyplot as plt

import numpy as np


def merge_nodes(nodes, edges, nodeIDclusters, debug=False):

    print(f'Before merging there are {len(nodes):,} nodes and {len(edges):,} edges')
    nNodes, nEdges = len(nodes), len(edges)
    if debug:
        origNodesIDs = nodes['nodeID'].unique()

    from_counts = edges['from'].value_counts()
    to_counts = edges['to'].value_counts()

    counts = pd.concat([from_counts,
                        to_counts],axis=1,sort=True).fillna(0).sum(axis=1).reset_index()
    counts.columns = ['nodeID','count']
    counts = counts.sort_values('count',
                                ascending=False)
    
    
    nodeMappings = {}
    for cluster in nodeIDclusters:
        cluster = list(cluster)
        
        cluster_counts = counts[counts['nodeID'].isin(cluster)]
        if len(cluster_counts) > 0:
            consolidatedID = cluster_counts['nodeID'].values[0]
        elif len(cluster) > 0:
            consolidatedID = cluster[0]
        else:
            continue
            
        for mappedID in cluster:
            if mappedID == consolidatedID:
                continue 
            nodeMappings[mappedID] = consolidatedID

            
    def updateNodeID(x, nodeMappings):
        if x in nodeMappings.keys():
            return nodeMappings[x]
        return x
    
    nodes['nodeID'] = nodes['nodeID'].apply(lambda x: updateNodeID(x, nodeMappings))
    for ecol in ['from','to']:
        edges[ecol] = edges[ecol].apply(lambda x: updateNodeID(x, nodeMappings)) 
        
    # drop duplicate edges
    edges = edges.reset_index(drop=True)
    edges = edges.iloc[edges.astype(str).drop_duplicates().index].copy().reset_index(drop=True)

    
    def combineFeat(x):
        x = list(set(x))
        if len(x) == 1:
            x = x[0]
        return x
    # consolidate nodes
    nodes = nodes.groupby(['nodeID','nodeType'],as_index=False).agg({c:lambda x: combineFeat(x) for c in 
                                                                     list(set(nodes.columns) - set(['nodeID','nodeType']))})
    nodes = nodes.reset_index(drop=True)
    
    print(f'After merging there are {len(nodes):,} nodes and {len(edges):,} edges')
    reducNodes = (nNodes - len(nodes)) 
    reducEdges = (nEdges - len(edges)) 
    reducNodesPCT = round(100 * reducNodes/nNodes,2)
    reducEdgesPCT = round(100 * reducEdges/nEdges,2)
    print('Reduction:')
    print(f'\t -{reducNodes} (-{reducNodesPCT}%)')
    print(f'\t -{reducEdges} (-{reducEdgesPCT}%)')
    
    if debug:
        print('---DEBUG MODE---')
        print('Nodes removed:')
        for i in list(set(origNodesIDs) - set(nodes['nodeIDs'])):
            print('\t"'+str(i)+'"')
    
    
    return nodes, edges, nodeMappings


import ipywidgets
from ipywidgets import Layout


def displayH3(x):
    display(HTML(f'<h3>{x}</h3>')) 
    
          

class interactiveMerger: 
    def __init__(self, nodes, edges, savedir, field, G_context=None, chunksize=10000, blacklist=None, candidate_thresh=0.9,
                 edge_attr=['edgeType'], saveprefix=None,removeDisplayNames=[],cleaningFunc=None, deduplicateGlobal=False,
                 verbose=1, removeFloatingSingletonNodes=False, randomseed=None, newIteration=False):
        
        if randomseed is not None: np.random.seed(randomseed) 

        self.resolveDisplay=nodes.set_index('nodeID').displayName.to_dict().get
        self.resolveNodeType=nodes.set_index('nodeID').nodeType.to_dict()  
        
        if removeFloatingSingletonNodes:
            linkedNodes = set(edges['from']).union(set(edges['to']))
            nodes = nodes[nodes['nodeID'].isin(linkedNodes)].copy()
        
        if deduplicateGlobal:
            # identify duplicates mapping and remove duplications from nodes dataframe
            displayNamesToRemove = removeDisplayNamesDefault+removeDisplayNames 
            nodes, updatedIDMapping = deduplication.mergeDuplicateNodes(nodes, 
                                                                        displayNamesToFilterOut=displayNamesToRemove) 
            edges = deduplication.mergeDuplicateEdges(edges, updatedIDMapping) 
            
            
        self.savedir = savedir
        self.savePrefix = saveprefix
        self.field = field
        self.blacklist = blacklist
        if self.blacklist is None:
            self.blacklist = os.path.join(self.savedir,'blacklist.csv')
        elif '/' not in self.blacklist:
            self.blacklist = os.path.join(self.savedir,self.blacklist)
        self.candidate_thresh = candidate_thresh
        self.chunksize = chunksize
        
        self.merger = None 
        self.comms = None 
        self.edges = edges
        
        self.nodesDataframes = {}
        insti_nodeData = nodes[nodes['nodeType']=='Institution'].copy()
        if len(insti_nodeData) > 0 and 'address' in insti_nodeData.columns:
            self.nodesDataframes['Institution'] = insti_nodeData[['nodeID',
                                                                  'address']].set_index('nodeID')
        
        # split nodes of interest and other nodes        
        self.otherNodes = nodes[nodes['nodeType']!=self.field].copy().reset_index(drop=True)
        nodes = nodes[nodes['nodeType']==self.field].copy().reset_index(drop=True)
        
        # Load/Create MergeHistory 
        mergeHistoryPath = os.path.join(self.savedir,f'mergeHistory-{self.field}.json')
        if os.path.exists(mergeHistoryPath):
            print('Loading mergeHistory..')
            self.mergeHistory = json.load(open(mergeHistoryPath,'r')) 
            
            lastIteration = str(max([int(k) for k in self.mergeHistory.keys()]))
            if 'chunkNodeLists' in self.mergeHistory[lastIteration].keys():
                ## do updated load
                chunkNodeLists = self.mergeHistory[lastIteration]['chunkNodeLists']
                lastChunk = max([int(x) for x in self.mergeHistory[lastIteration]['chunkNodeLists'].keys()])
                
                lastActiveChunkID = max([int(x) for x in list(self.mergeHistory[str(lastIteration)]['chunkHistory'].keys())]) 
                lastChunkMerged = ('collapsedNodes' in self.mergeHistory[lastIteration]['chunkHistory'][str(lastActiveChunkID)].keys())   
                
                if newIteration or (str(lastActiveChunkID) == str(lastChunk) and lastChunkMerged):
                    self.iteration = str(int(lastIteration) + 1)
                    self._setupNodeChunks(nodes) 
                elif int(lastActiveChunkID) < int(lastChunk) and lastChunkMerged:
                    self.iteration = str(lastIteration)
                    self.activeChunkID = str(lastActiveChunkID + 1)
                    self._reloadNodeChunks(nodes, chunkNodeLists, self.activeChunkID) 
                elif int(lastActiveChunkID) < int(lastChunk) and not lastChunkMerged:
                    self.iteration = str(lastIteration)
                    self.activeChunkID = str(lastActiveChunkID)
                    self._reloadNodeChunks(nodes, chunkNodeLists, self.activeChunkID)   
            else:
                # older mergeHistory, before the update to save chunk nodeID lists for reload
                self.iteration = str(int(max([int(k) for k in self.mergeHistory.keys()])) + 1)
                self._setupNodeChunks(nodes)  
        else:
            self.mergeHistory = {}
            self.iteration = '0'
            self._setupNodeChunks(nodes)
            
        print('Starting Iteration {}..'.format(self.iteration))
        self._instantiateMerger() 
         
        self.edges = edges
        self.canMerge = False
       
        self._summarize(len(nodes))
         
        ##############################
            
        self.iteration = str(self.iteration)
        self.widget = None
        if G_context is None:
            G_context = nx.from_pandas_edgelist(edges, source='from', target='to', edge_attr=edge_attr)
        self.G_context = G_context
               
        
    def _summarize(self, nNodes):
        summaryTemplate = """<br>
                             <b>Field:</b> {} <br>
                             <b>Iteration:</b> {} 
                             <p>{:,} Nodes ({:,} Chunks of {:,} Nodes) <br> {:,} Edges </p>
                             <hr>
                             <h3>Active ChunkID = {}</h3>
                             <hr>"""
        display(HTML(summaryTemplate.format(self.field, self.iteration, nNodes, 
                                            len(self.nodes.keys()), self.chunksize,
                                            len(self.edges), self.activeChunkID)))

          
    def updateField(self, newField, chunksize=None, candidate_thresh=None, blacklist=None, savePrefix=None): 
        self.save()
        if chunksize is None: chunksize = self.chunksize
        if blacklist is None: blacklist = self.blacklist
        if candidate_thresh is None: candidate_thresh = self.candidate_thresh
        if savePrefix is None: savePrefix = self.savePrefix
        nodes = self._allNodes()
        self = interactiveMerger(nodes, self.edges, self.savedir, newField, G_context=self.G_context, 
                                chunksize=chunksize, blacklist=blacklist, 
                                candidate_thresh=candidate_thresh, saveprefix=savePrefix)
                  
        return(self)
    
    @lru_cache(100)
    def getConfig(self):
        "Returns the entityMerger config"
        if self.field == 'Scientist':
            config = {'text_field':'displayName',
                      'id_field':'nodeID',
                      'node_type':'Scientist',
                      'steps':{
                                'displayName':['lev','tfidf','parts','initials','likelihood']
                      }} 
        elif self.field in ['Institution','Location','Source',
                           'Conference','Book','Journal']:
            config = {'text_field':'displayName',
                      'id_field':'nodeID',
                      'node_type':self.field,
                      'steps':{
                                'displayName':['lev','tfidf']
                      }}
        return config
    
    def _allNodes(self):
        return pd.concat([self.otherNodes]+list(self.nodes.values()),sort=False).reset_index(drop=True) 
    def _nodes(self):
        return pd.concat(list(self.nodes.values())).reset_index(drop=True) 
        
    def _instantiateMerger(self):  
        if self.merger is None: 
            self.merger = EntityMerger(self.nodes[self.activeChunkID], self.edges, 
                                       config=self.getConfig(),
                                       candidate_thresh=self.candidate_thresh, 
                                       blacklist=self.blacklist)
            
        
    def runTextSimilarity(self):
        "Calculate text similarity for nodes in active chunk"
        if self.text_sims is None:
            print('{} Nodes in Chunk {}'.format(len(self.nodes[self.activeChunkID]), self.activeChunkID))
            if self.merger is None: 
                self._instantiateMerger()  
                
            start = datetime.datetime.now() 
            #perform text similarity comparisons
            self.text_sims = self.merger.get_text_similarities() 

            print('\n', datetime.datetime.now() - start) 
            
    def runGraphSimilarity(self): 
        "Calculate graph-based similarity for nodes in active chunk"
        if self.graph_sims is None:
            print('{} Nodes in Chunk {}'.format(len(self.nodes[self.activeChunkID]), self.activeChunkID))
            if self.merger is None: 
                self._instantiateMerger()  
                
            start = datetime.datetime.now() 
            #perform text similarity comparisons
            self.graph_sims = self.merger.get_graph_sims() 

            print('\n', datetime.datetime.now() - start) 
            
    def _run_similarity_for_nodes(self):
        if self.nodes[self.activeChunkID].shape[0] > 0: 
            self.nodes[self.activeChunkID] = self.nodes[self.activeChunkID].explode('displayName').fillna('')
            self._instantiateMerger()
            self.runTextSimilarity()
            self.runGraphSimilarity() 
        
    def _setupNodeChunks(self, nodes): 
        print('setting up chunks for new iteration...')
        def shuffleRows(df): 
            df = df.reset_index(drop=True)
            arr = np.arange(len(df))
            out = np.random.permutation(arr) # random shuffle 
            return df.iloc[out].reset_index(drop=True) 
        # shuffle rows of dataframe
        print('shuffle rows...')
        nodes = shuffleRows(nodes)
        # chunk nodes
        nChunks = math.ceil(len(nodes)/self.chunksize) 
        self.nodes = {str(i):nodes.iloc[i*self.chunksize:(i+1)*self.chunksize] for i in range(nChunks)}
        del nodes
        # instantiate flags
        self.nodesMerged = {i:False for i in self.nodes} 
        self.canContinue = False  
        self.activeChunkID = '0'
        self.mergeHistory[self.iteration] = {'chunkHistory':{}}
        # preserve record of which nodes are in each chunk
        chunkNodeLists = {i: list(self.nodes[i].nodeID.unique()) for i in self.nodes.keys()}
        self.mergeHistory[self.iteration]['chunkNodeLists'] = chunkNodeLists
        # reset variables  
        self.threshold = None
        self.simCol = None
        self.text_sims = None
        self.graph_sims = None 
        self._run_similarity_for_nodes() 
        
        
    def _reloadNodeChunks(self, nodes, nodeLists, activeChunkID): 
        print('reloading chunks for in-progress iteration')
        self.nodes = {str(i):nodes[nodes['nodeID'].isin(nodeLists[i])].copy() for i in nodeLists.keys()}
        del nodes
        # instantiate flags
        self.nodesMerged = {i:(int(i) < int(activeChunkID)) for i in self.nodes} 
        self.canContinue = False  
        self.activeChunkID = str(activeChunkID)  
        # reset variables  
        self.threshold = None
        self.simCol = None
        self.text_sims = None
        self.graph_sims = None 
        self._run_similarity_for_nodes() 
        
    def simScatter(self):
        plt.figure(figsize=(4,4))
        plt.scatter(self.graph_sims.graph_sim_total, self.graph_sims.text_sim_total)
        _ = plt.xlabel('graph_sim_total')
        _ = plt.ylabel('text_sim_total')
        plt.show()
        
    def simHist(self, simCol=None, thresh=None, grid=False, textOnly=False): 
        "Plot distribution of similarity, specified in simCol (default is 'text_sim_total')." 
        if textOnly:
            simdf = self.text_sims
            if simCol is None: simCol ='text_sim_total'
        else:
            simdf = self.graph_sims
            if simCol is None: simCol ='sim_total' 
        if thresh is not None:
            simdf = simdf[simdf[simCol]>thresh].copy() 
        simdf[simCol].hist(grid=grid)
        plt.show() 
        
    def resolve(self, metric='text_sim_total', constraints=None,
                removeDisplayNames=[], cleaningFunc=None, widgetStateString=None,
                deduplicate=False):
        self.kwargs = None
        
        if constraints is None:
            if self.field == 'Scientist':
                constraints={'text_sim_total':2,'graph_sim_total':0}
            else:
                constraints={'text_sim_total':1,'graph_sim_total':0}
                
        display(HTML(f'<b>Metric:</b> {metric} <br><b>Constraints</b>: {constraints}'))
        
        nodes = self.nodes[self.activeChunkID] 
        for c in nodes.columns:
            if len(nodes[c].dropna()) == 0:
                del nodes[c]
            nodes[c] = nodes[c].fillna('')
            if len([x for x in nodes[c] if type(x) is list]) > 0:
                nodes = nodes.explode(c)
            if set(nodes[c].unique()) == set(['']):
                del nodes[c]
         
        removeDisplayNames = removeDisplayNames + removeDisplayNamesDefault
        
        # setup mergeHistory
        self.mergeHistory[self.iteration]['chunkHistory'][str(self.activeChunkID)] = {'metric':metric,'constraints':constraints,
                                                                 'removeDisplayNames':removeDisplayNames,
                                                                 'nNodes':{'Before Merging':
                                                                           self.nodes[self.activeChunkID].nodeID.nunique()},
                                                                 'nEdges':{'Before Merging':
                                                                           len(self.edges)}} 
         
        
        origNodeIDs = self.nodes[self.activeChunkID]['nodeID'].unique() 
        if deduplicate:
            # deduplicate nodes and edges
            self.nodes[self.activeChunkID], self.edges, ngb = deduplication.removeDuplicates(nodes, self.edges, 
                                                                               removeDisplayNames=removeDisplayNames,
                                                                               cleaningFunc=cleaningFunc,
                                                                               verbose=0) 
        else: 
            origNodeIDs = self.nodes[self.activeChunkID]['nodeID'].unique() 
            self.nodes[self.activeChunkID] = self.nodes[self.activeChunkID][
                ~self.nodes[self.activeChunkID]['displayName'].isin(removeDisplayNames)
            ].copy()  
            
        removeNodeIDs = set(origNodeIDs) - set(self.nodes[self.activeChunkID]['nodeID'].unique() )
        self.edges = self.edges[~self.edges['from'].isin(removeNodeIDs)].copy()
        self.edges = self.edges[~self.edges['to'].isin(removeNodeIDs)].copy()
            
        # get args for widget
        G = self.getMergedGraph(metric=metric, constraints=constraints, removeIDs=removeNodeIDs)                                                                                                                                              
        self.kwargs = dict(
            G=G, 
            G_context=self.G_context,
            weight=metric,
            display=self.resolveDisplay, 
            nodeType=self.resolveNodeType, 
            context_radius=1 + (self.field == 'Scientist')
        )            
        
        if self.field == 'Institution':
            self.kwargs['nodeData'] = self.nodesDataframes[self.field]
                                                                               
        
        #self.mergeHistory[self.iteration]['resolveState'][self.activeChunkID] = {'constraints':constraints,
        #                                                                         'kwargs':self.kwargs}
        
        if len(list(G.edges())) == 0:
            displayH3('No potential pairs to resolve given constraints')

        try:
            # instatiate widget
            if widgetStateString is None:
                self.widget = IdentityMatcher(**self.kwargs) 
            else: 
                accepted = json.loads(widgetStateString) 
                self.widget = IdentityMatcher(**self.kwargs, accepted=accepted)
        except Exception as e:
            print(str(e))
            print('\n\n Do you have expertwidgets-0.1.0b7-py2.py3-none-any.whl or later installed?')
        
        return self.widget
        

    def nextChunk(self):
        "Advances to the next chunk of nodes to merge"
        if not self.canContinue:
            verification = input("No nodes have been merged, are you sure you want to continue to the next chunk? (y|n): ")
            if verification.lower() in ['y','yes']:
                self.canContinue
        if self.canContinue: 
            self.activeChunkID = str(int(self.activeChunkID) + 1)
            self.merger = None
            self.text_sims = None
            self.graph_sims = None
            self.comms = None
            if self.activeChunkID not in self.nodes:
                self.activeChunkID = None
            else:
                display(HTML(f'<h4>Active Chunk is now Chunk {self.activeChunkID} </h4>'))
                #print('Active Chunk is now Chunk '+str(self.activeChunkID))
        else:
            display(HTML(f'<h4>Did not continue to next chunk. Active Chunk = {self.activeChunkID} </h4>'))
            #print('Did not continue to next chunk.\nActive Chunk = '+str(self.activeChunkID))
            
            
    def _complete_merge(self, strong=True, weak=False, skipMerge=False, debug=False):
        def merging(self, clustersToMerge, debug):
            
            self.nodes[self.activeChunkID], self.edges, nodeMapping = merge_nodes(self.nodes[self.activeChunkID], 
                                                                                  self.edges, clustersToMerge, debug=debug)
            # update mergeHistory
            activeChunkIDKey =str(self.activeChunkID)
            self.mergeHistory[self.iteration]['chunkHistory'][activeChunkIDKey]['collapsedNodes'] = nodeMapping 
            self.mergeHistory[self.iteration]['chunkHistory'][activeChunkIDKey]['nNodes']['After Merging'] = len(self.nodes[self.activeChunkID])
            self.mergeHistory[self.iteration]['chunkHistory'][activeChunkIDKey]['nEdges']['After Merging'] = len(self.edges)
            print('\n\n\n')
            
        def postMergeActions(self):
            self.canContinue = True
            self.nextChunk()
            if self.activeChunkID is not None:
                self._run_similarity_for_nodes()  
            else:
                displayH3('All chunks have been merged!')
                display('Do you want to call .nextIteration() to strart a new iteration?')
                
            
        self.merge_strong = None
        self.merge_weak = None 
        
        if not weak and skipMerge:
            displayH3('Continuing to the next chunk without merging....')
        else:
            if weak and len(self.widget.weak_groups) > 0:
                displayH3('Merging weak groups from Identity Matcher...') 
                
            clustersToMerge = []
            if self.widget is None:
                displayH3('Hmmn, nothing to merge.. did you resolve any entities? (use .resolve() to launch widget)')
                return
            if strong:
                clustersToMerge.extend(self.widget.strong_groups)
            if weak: 
                clustersToMerge.extend(self.widget.weak_groups)
                
            if debug:
                display(HTML('<i> ---DEBUG MODE---- Confirm merge for the specified clusters: </i>'))
                for i in clustersToMerge:
                    display(str(i)+'\n')
                    
            
                buttonCancel = ipywidgets.Button(description='Cancel') 
                def cancel(_):   
                    displayH3('Merge cancelled') 
                buttonCancel.on_click(cancel) 

                buttonContinue = ipywidgets.Button(description='Continue') 
                def approved(_):   
                    merging(self, clustersToMerge, debug)
                    postMergeActions(self)

                buttonContinue.on_click(approved) 

                buttonsWidget = ipywidgets.HBox((buttonCancel, buttonContinue)) 
                return buttonsWidget
        
            else:
                merging(self, clustersToMerge, debug) 
            
        postMergeActions(self) 
            
            
    def merge(self, strong=True, weak=False, forceMerge=False, widgetStateString=None, debug=False): 
        if widgetStateString is not None:
            print(widgetStateString)
            accepted = json.loads(widgetStateString) 
            self.widget = IdentityMatcher(**self.kwargs, accepted=accepted)
            display(self.widget)
            clear_output()
            print('self.widget.strong_groups:',self.widget.strong_groups)
            self.merge(strong=strong, weak=weak, forceMerge=forceMerge)
        
        if self.widget is None:
            print('Hmmn, nothing to merge.. did you resolve any entities? (use .resolve() to launch widget)')
            return
    
        self.merge_strong = strong
        self.merge_weak = weak
        
        if forceMerge:
            return self._complete_merge(strong=self.merge_strong,weak=self.merge_weak, debug=debug) 
            
        if len(self.widget.strong_groups) == 0: 
            clear_output()
            display(HTML('<h3> There are no node sets in the "strong groups" from the IdentityMatcher, do you want to cancel the merge? </h3> Note: If you accepted/rejected any pairs in the IdentifyMatcher widget, you should cancel this merge and pass the accept/reject results from the widget using the copy to clipboard button otherwise your accept/reject work will be lost!'))
            
            buttonCancel = ipywidgets.Button(description='Cancel') 
            def cancel(_):  
                clear_output()
                displayH3('Merge cancelled') 
            buttonCancel.on_click(cancel) 

            buttonContinueWithout = ipywidgets.Button(description='Continue') 
            def continueWithout(_):  
                clear_output() 
                self._complete_merge(strong=self.merge_strong,weak=self.merge_weak, skipMerge=True, debug=debug) 

            buttonContinueWithout.on_click(continueWithout) 

            buttonsWidget = ipywidgets.HBox((buttonCancel, buttonContinueWithout)) 
            return buttonsWidget
        
        else:
            self._complete_merge(strong=strong, weak=weak, debug=debug)

    
    
    
    def nextIteration(self, chunksize=None):
        "Resets to next Iteration of merging"
        if chunksize is not None: self.chunksize = chunksize
        nodes = self._nodes() 
        self.iteration = str(int(self.iteration) + 1)
        self._setupNodeChunks(nodes) 
        print('- Reset Complete -')
        self._summarize(len(nodes))
        
    def getMergedGraph(self, metric='sim_total', constraints={}, removeIDs=None): 
        def edgeRep(fromNode, toNode):
            return f'{fromNode}-->{toNode}'
        
        simsDF = self.merger.text_sims[~self.merger.text_sims['blacklist']].copy()
        
        if removeIDs is not None:
            simsDF = simsDF[~simsDF['from'].isin(removeIDs)].copy()
            simsDF = simsDF[~simsDF['to'].isin(removeIDs)].copy()
        
        simsDF = simsDF[simsDF['from']!=simsDF['to']].copy()
        simsDF['edgeRep'] = simsDF.apply(lambda x: edgeRep(x['from'],x['to']), axis=1)
        
        edgeDF = simsDF.copy() 
        keep = set(edgeDF['edgeRep'])
        #print(metric, 0, len(keep)) 
        for constraintMetric in constraints.keys():
            thresh = constraints[constraintMetric]
            #if self.merger.text_sims
            constraintEdgeDF = simsDF[simsDF[constraintMetric] > thresh].copy() 
            keep = keep.intersection(set(constraintEdgeDF['edgeRep']))
            #print(constraintMetric, thresh, len(keep))
        
        edgeDF = edgeDF[edgeDF['edgeRep'].isin(keep)].copy().drop(columns=['edgeRep'])
        return nx.from_pandas_edgelist(edgeDF, 'from', 'to', metric)
        
    def save(self,append=False):  
        "Save all nodes, edges, mergeHistory, and delete lists to savedir specified on instantiation"
        os.makedirs(self.savedir, exist_ok=True)
        if append:
            writemode='a'
        else:
            writemode='w'
        if self.savePrefix != '' and self.savePrefix[-1] not in ['_','-']:
            fnPrefix = f'{self.savePrefix}_'
        else:
            fnPrefix = self.savePrefix
        display(HTML('Saving to <b>{}</b>\n'.format(self.savedir)))
        # save all nodes!
        nodes = self._allNodes()
        print('\tSaving Nodes...')
        for nodetype, ndf in nodes.groupby('nodeType'): 
            for c in ndf.columns:
                if len(ndf[c].dropna())==0: del ndf[c]
            print('\t\t'+f'{nodetype} - {ndf.shape[0]:,}')
            with open(os.path.join(self.savedir,f'{fnPrefix}Nodes_{nodetype}.jsonl'),writemode) as f:
                f.write(ndf.to_json(lines=True, orient='records')+'\n') 

        print('\tSaving Edges...')
        for edgetype, edf in self.edges.groupby('edgeType'): 
            for c in edf.columns:
                if len(edf[c].dropna())==0: del edf[c]
            print('\t\t'+f'{edgetype} - {edf.shape[0]:,}')
            with open(os.path.join(self.savedir,f'{fnPrefix}Links_{edgetype}.jsonl'),writemode) as f:
                f.write(edf.to_json(lines=True, orient='records')+'\n') 

        # save merge history
        print('\tSaving Merge History...')
        with open(os.path.join(self.savedir,f'mergeHistory-{self.field}.json'),'w') as f:
            json.dump(self.mergeHistory, f, sort_keys=True, indent=4)  
            
            
            