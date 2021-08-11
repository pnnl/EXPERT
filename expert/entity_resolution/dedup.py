import pandas as pd
from IPython.display import display, HTML
import datetime, re
from unidecode import unidecode
import networkx as nx

from tqdm import tqdm


removeDisplayNamesDefault = ['research_orgs_','contributor_orgs_', 
                               'Materials Project', 'None None', 'and others', ]


NODETYPES = ['Scientist','Source','Journal','Conference','Book','Paper','Location',
                'Institution','Funding Agency','Topic']
EDGETYPES = ['Author Of','Belongs To','Published In','Related To','Located In','Belongs To','Funded']


def typeSummary(df, typecol):
    vc = pd.DataFrame(df[typecol].value_counts().items(), columns=['type','count']).set_index('type')
    vc['count'] = vc['count'].apply(lambda x:format(x,','))
    display(vc)
    
class deduplication:

        
    def updateNodeIDs(df, updatedIDMapping, id_column='nodeID', filterOutIDs=[]):
        filterOutIDs=filterOutIDs+['REMOVE ME','']
        unchangedNodeIDs = set(df[id_column].unique()) - set(updatedIDMapping.keys())
        replacementMapping = updatedIDMapping
        replacementMapping.update({x:x for x in unchangedNodeIDs})
        df = df.fillna('').drop_duplicates(subset=id_column)
        df[id_column] = df[id_column].map(replacementMapping)
        df = df[~df[id_column].isin(filterOutIDs)].copy()
        return df
        
    def identifyDuplicateNodes(nodes):
        def updateNodeID(x):
            return re.sub(' +',' ', re.sub('[.,!?$]',' ', str(x).lower())  ).strip() 
        #for i in nodes.columns:
        #    nodes = nodes.explode(i)
        ngb = nodes.fillna('').assign(displayNameLower=nodes.displayName.apply(lambda x:updateNodeID(x))) 
        gbcols = list(set(ngb.columns)-set(['nodeID','displayName']))
        for c in ngb.columns: 
            if list in set([type(x) for x in ngb[c]]): ngb = ngb.explode(c)
        #ngb = ngb.explode('displayName')
        ngb = ngb.groupby(gbcols).agg({'nodeID':list,'displayName':list})
        ngb['nIDs'] = ngb['nodeID'].apply(lambda x:len(x))
        ngb['consolidatedNodeID'] = ngb['nodeID'].apply(lambda x:sorted(x)[0])
        return ngb
        
    def identifyDuplicateMappings(nodes, filterOutIDs=removeDisplayNamesDefault, id_column='nodeID'):
        updatedIDMapping = {x:'REMOVE ME' for x in filterOutIDs}

        nodes = nodes[~nodes['nodeID'].isin(filterOutIDs)]
        nodes = nodes.fillna('').drop_duplicates(subset=id_column).copy()

        ngb = deduplication.identifyDuplicateNodes(nodes)
        dupls = ngb[ngb['nIDs']>1].copy()

        for consolidatedNodeID, keys in zip(dupls['consolidatedNodeID'],dupls['nodeID']):
            updatedIDMapping.update({k:consolidatedNodeID for k in keys}) 
        return updatedIDMapping 
    
    def combineToSingleDF(dfs):
        dfs = [df for df in dfs if df is not None]
        try:
            return pd.concat(dfs,sort=False)  
        except:
            return None

    def _load_df(itemtypes, datadir, verbose=True):
        def _filepaths(itemtypes, datadir, verbose=True):
            if verbose: print(itemtypes, datadir)
            fp = []
            for itype in itemtypes: 
                fp.extend(glob.glob(os.path.join(datadir,'*'+itype.replace('_','*').replace(' ','*')+'*.jsonl')))
            if verbose:
                for f in fp: print('\t'+f)
            return fp
        fp = _filepaths(itemtypes, datadir, verbose=True)
        if len(fp)==0:
            data = None
        elif len(fp) == 1:
            data = pd.read_json(fp[0],lines=True)
        else:
            data = [pd.read_json(f,lines=True) for f in tqdm(fp)] 
            if verbose:  
                print('combining...') 
            data = combineToSingleDF(data)
        return data 

    def mergeDuplicateNodes(nodes, displayNamesToFilterOut=removeDisplayNamesDefault): 
        now = datetime.datetime.now()
        nodes = nodes.fillna('').drop_duplicates(subset='nodeID')
        nodes = nodes[nodes['nodeID']!=''].copy()

        print('Generate duplicate nodeID mapping to consolidate duplicates...')
        filterOutIDs_basic = nodes[nodes['displayName'].isin(displayNamesToFilterOut)]['nodeID'].unique()

        updatedIDMapping = deduplication.identifyDuplicateMappings(nodes, filterOutIDs=filterOutIDs_basic)

        nodes = deduplication.updateNodeIDs(nodes, updatedIDMapping, id_column='nodeID', filterOutIDs=['REMOVE ME',''])

        print(datetime.datetime.now() - now)
        typeSummary(nodes,'nodeType')

        return nodes, updatedIDMapping
    
    
    def mergeDuplicateNodesFromDir(datasource,
                     displayNamesToFilterOut=removeDisplayNamesDefault,
                     datadir='./',
                     nodetypes=NODETYPES): 
        now = datetime.datetime.now()
        savedir = f'{datasource}_deduplicated'

        print(f'Loading data for {datasource}...')
        if os.path.exists(savedir):# is not None:
            print('Attempting to load from savedir...')
            nodes = _load_df(nodetypes, savedir, verbose=True) 

            if nodes is not None: 
                print('\tLoaded nodes: ', set(nodes.nodeType.unique()))
                missingNodeTypes = list(set(nodetypes)-set(nodes.nodeType.unique()))
            else:
                missingNodeTypes = nodetypes 

            if len(missingNodeTypes) > 0:
                print('Savedir is missing nodeTypes: ', missingNodeTypes)

                print('Attempting to load from datadir...')
                if len(missingNodeTypes) > 0: 

                    nodes2 = _load_df(nodetypes, datadir, verbose=True)
                    nodes = combineToSingleDF([nodes,nodes2]) 

        else:
            print(f'{savedir} does not exist.\nLoading from datadir...')
            nodes = _load_df(nodetypes, datadir, verbose=True) 
 
        print('Nodes data loaded!') 
        nodes, updatedIDMapping = deduplication.mergeDuplicateNodes(nodes, displayNamesToFilterOut=removeDisplayNamesDefault)
        return nodes, updatedIDMapping


    def mergeDuplicateEdges(edges, updatedIDMapping): 
        now = datetime.datetime.now() 
        edges = edges.fillna('').drop(columns=['index','_id'],
                                      errors='ignore').drop_duplicates().dropna(subset=['from','to']) 
         
        edges = deduplication.updateNodeIDs(edges, updatedIDMapping, id_column='from', filterOutIDs=['REMOVE ME',''])
        edges = deduplication.updateNodeIDs(edges, updatedIDMapping, id_column='to', filterOutIDs=['REMOVE ME',''])
            
        print(datetime.datetime.now() - now)
        typeSummary(edges,'edgeType')

        return edges


    def mergeDuplicateEdgesromDir(datasource, updatedIDMapping,
                          datadir='./',
                          edgetypes=EDGETYPES): 
        now = datetime.datetime.now()
        savedir = f'{datasource}_deduplicated'

        if os.path.exists(savedir): 
            print('Attempting to load edges from savedir...') 
            edges = _load_df(edgetypes, savedir, verbose=True) 

            if edges is not None: 
                print('\tLoaded edges: ', set(edges.edgeType.unique()))
                missingEdgeTypes = list(set(edgetypes)-set(edges.edgeType.unique()))
            else:
                missingEdgeTypes = edgetypes

            if len(missingEdgeTypes) > 0:
                print('Savedir is missing edgeTypes: ', missingEdgeTypes)
                print('Attempting to load from datadir...')
                if len(missingEdgeTypes) > 0:
                    edges2 = _load_df(missingEdgeTypes, datadir, verbose=True) 
                    edges =combineToSingleDF([edges,edges2]) 
        else:
            print(f'{savedir} does not exist.')
            edges = _load_df(edgetypes, datadir, verbose=True) 

        print('Edge data loaded!')
        edges = deduplication.mergeDuplicateEdges(edges, updatedIDMapping)

        return edges


    def saveDeduplicated(datasource, nodes, edges, append=True, savedir=None):
        if savedir is None: savedir = f'./{datasource}_deduplicated/'
        os.makedirs(savedir, exist_ok=True)
        print(savedir)
        if append:
            writemode='a'
        else:
            writemode='w'
        if nodes is not None:
            for nodetype, ndf in nodes.groupby('nodeType'):
                print(nodetype)
                for c in ndf.columns:
                    if len(ndf[c].dropna())==0: del ndf[c]
                print('\t'+f'{ndf.shape[0]:,}')
                with open(os.path.join(savedir,f'{datasource}_Nodes_{nodetype}.jsonl'),writemode) as f:
                    f.write(ndf.to_json(lines=True, orient='records')+'\n') 

        if edges is not None:
            for edgetype, edf in edges.groupby('edgeType'):
                print(edgetype)
                for c in edf.columns:
                    if len(edf[c].dropna())==0: del edf[c]
                print('\t'+f'{edf.shape[0]:,}')
                with open(os.path.join(savedir,f'{datasource}_Links_{edgetype}.jsonl'),writemode) as f:
                    f.write(edf.to_json(lines=True, orient='records')+'\n') 



    ########################################################################################################    
    
        

    def deleteNodesWithDisplayName(nodes, displayNamesToRemove, verbose=2):
        if verbose > 1:
            print('Removing:')
            for i in displayNamesToRemove: print('\t'+i)
        nodes2 = nodes[~nodes['displayName'].isin(displayNamesToRemove)].copy()
        removedN = nodes.shape[0] - nodes2.shape[0] 
        removedPCT = 100*removedN/nodes.shape[0]
        if verbose > 1: print('This removed {:.2f}% ({:,} nodes)'.format(removedPCT,removedN))
        if verbose > 0: print('After removal based on removeDisplayNames displayName values, there are {:,} nodes\n'.format(len(nodes2))) 
        return nodes2

    def identifyDuplicateNodes2(nodes):
        def updateNodeID(x):
            return re.sub(' +',' ', re.sub('[.,!?$]',' ', x.lower())  ).strip() 
        for i in nodes.columns:
            nodes = nodes.explode(i)
        ngb = nodes.fillna('').assign(displayNameLower=
                                      nodes.displayName.apply(lambda x:updateNodeID(x))) 
        gbcols = list(set(ngb.columns)-set(['nodeID','displayName']))
        ngb = ngb.groupby(gbcols).agg({'nodeID':list,'displayName':list})
        ngb['nIDs'] = ngb['nodeID'].apply(lambda x:len(x))
        ngb['consolidatedNodeID'] = ngb['nodeID'].apply(lambda x:sorted(x)[0])
        return ngb

    def mergeDuplicates(mergeData, nodes, edges, oldKeysCol='matches', newKeyCol='mapTo', verbose=2): 
        start = datetime.datetime.now()
        nodes, edges = nodes.copy(), edges.copy() 
        if verbose > 1: print(f'Before merging duplicates there are {len(nodes):,} nodes and {len(edges):,} edges')

        mergeData = mergeData.explode(oldKeysCol)
        allOtherNodes = list(set(edges['from']).union(edges['to']) - set(mergeData[oldKeysCol]))
        mergeMap = {oldKey:newKey for oldKey,newKey in zip(list(mergeData[oldKeysCol])+allOtherNodes,
                                                           list(mergeData[newKeyCol])+allOtherNodes)}

        edges['from'] = edges['from'].map(mergeMap)#.fillna(edges['from'])
        edges['to'] = edges['to'].map(mergeMap)#.fillna(edges['to'])
        nodeIDs = list(set(edges['from']).union(set(edges['to'])))
        nodes = nodes[nodes['nodeID'].isin(nodeIDs)].copy().reset_index(drop=True)

        if verbose > 1: print(f'After merging duplicates there are {len(nodes):,} nodes and {len(edges):,} edges')
        return nodes, edges

    def removeDuplicates(nodes, edges, removeDisplayNames=removeDisplayNamesDefault, cleaningFunc=None, verbose=2): 


        # remove exact duplicates
        nodes = nodes.fillna('').drop_duplicates(subset='nodeID')
        edges = edges.fillna('').drop_duplicates()

        # clean nodesDisplayNames with passed function, if specified
        if cleaningFunc is not None:
            nodes['displayName'] = nodes['displayName'].apply(lambda x: cleaningFunc(x))

        # remove nonsense nodes
        nodes = deduplication.deleteNodesWithDisplayName(nodes, removeDisplayNames, verbose=verbose)  
        if verbose > 1: deduplication.typeSummary(nodes,'nodeType')
        #print(nodes.nodeType.value_counts())

        nodesIDs = nodes['nodeID'].unique() 

        if verbose > 1: print(f'\nReduces to {len(edges):,} edges:')
        if verbose > 1: deduplication.typeSummary(edges,'edgeType')
        #print(edges.edgeType.value_counts())

        # identify duplicate nodes
        if verbose > 0: print('\n\nIdentifying duplicate nodes..')
        ngb = deduplication.identifyDuplicateNodes(nodes)
        dupls = ngb[ngb['nIDs']>1].copy()
        if verbose > 0: print(f'{dupls.shape[0]:,} duplicates')
        if verbose > 1: display(ngb.sort_values(by=['nIDs'],ascending=False).drop(columns=['nodeID']).head(10))

        if verbose > 0: print('\n\nConsolidating duplicate nodes..')
        now = datetime.datetime.now()
        nodes, edges = deduplication.mergeDuplicates(ngb, nodes, edges, oldKeysCol='nodeID', 
                                                     newKeyCol='consolidatedNodeID', verbose=verbose)
        if verbose > 1: print(datetime.datetime.now() - now)
        if verbose > 1: print(f'{len(nodes):,} nodes and {len(edges):,} edges\n') 

        if verbose > 0: print('Fill NANs, drop duplicates, drop any edges where from/to nodeID is a NAN..')

        # remove exact duplicates
        nodes = nodes.fillna('').drop_duplicates(subset='nodeID')
        edges = edges.fillna('').drop(columns=['index','_id'],errors='ignore').drop_duplicates().dropna(subset=['from','to']) 

        if verbose > 0: deduplication.typeSummary(nodes,'nodeType')

        if verbose > 0: deduplication.typeSummary(edges,'edgeType')

        return nodes, edges, ngb