import json
import datetime
import time
import os
import pandas as pd
import re
import numpy as np



NodeRepresentationsToDrop = ['{}','','[]',
                             "{'institutions': [{}]}","{'institutions': [], 'orcid': ''}",
                             "{'orcid': '', 'institutions': []}"]
            
class metadataGraph:
    """Format extracted links into the structure required for generating/visualizing the metadata graph"""  
    def _sourceType(row):
        source_type_mapping = {'Conference Proceeding':'Conference', 
                               'Journal':'Journal', 'Trade Journal':'Journal', 
                               'Book Series':'Book', 'Book':'Book', 'Multi-volume Reference Works':'Book',
                               '':'Source'}
        if row.nodeType == 'Conference/Journal/Source':
            if 'source_type' in row.node and row.node['source_type'] in source_type_mapping:
                return source_type_mapping[row.node['source_type']]
            print(row.node)
            return row.nodeType
        elif row.nodeType == 'Topic':
            return 'Topic/Expertise'
        else:
            return row.nodeType

    def _extract(x,f):
        if x is None or x == np.nan:return ''
        if f in x:
            return x[f]
        else:
            return ''
    def _country(x):
        spl = [y for y in x.split(',') if y.strip() != '']
        if len(spl) > 0:
            if re.sub(r'\d+', '', spl[-1]).strip() == '': return '' 
            return spl[-1].replace('Russian Federation','Russia').strip()
        else:
            return ''
    
    def _listify(x):
        if type(x) is list: return x
        return [x]
    
    def _getScientist_ID(x):
        try:
            return x['node']['origin_id']
        except:
            #print(x['node'])
            try:
                return 'Scientist'+str(x['node']['name'])
            except:
                print('\n\n\n',x['node'],'\n')
                return 'REMOVE ME'
                #return 'Scientist'+str(x['node']['name'])

    
    def _nodesDataframe(ldf):
        
        ### create node dataframes

        nodes = pd.concat([ldf[['from','fromNodeType']].copy().rename(columns={'from':'node','fromNodeType':'nodeType'}),
                           ldf[['to','toNodeType']].copy().rename(columns={'to':'node','toNodeType':'nodeType'})])
        nodes['nodeType'] = nodes.apply(metadataGraph._sourceType,axis=1) 
         
        nodes['node'] = nodes['node'].apply(metadataGraph._listify) 
        nodes = nodes.explode('node').dropna(subset=['node'])
        
        nodes = nodes[~nodes['node'].astype(str).isin(NodeRepresentationsToDrop)].copy()

        #nodes = nodes[~nodes['node'].astype(str).contains("'name'")].copy()
        
        
        # Papers
        papers = nodes[nodes['nodeType']=='Paper'].copy()
        
        papers['str_node'] = papers['node'].apply(lambda x: type(x) is str)
        
        strPapers = papers[papers['str_node']].copy()
        if len(strPapers) > 0:
            strPapers['nodeID'] = strPapers['node']
            strPapers['displayName'] = strPapers['nodeID']
        
        papers = papers[~papers['str_node']].copy()       
        if len(papers) > 0: 
            for k in ['origin','origin_id','doi','title','abstract','_id']:
                papers[k] = papers.apply(lambda x:x['node'].get(k), axis=1) 
            papers['nodeID'] = papers['origin'] + papers['origin_id']
            papers['displayName'] = papers['origin'] + papers['origin_id'] 
        else:
            papers = pd.DataFrame(columns=['nodeID','displayName'])

        
        
        if len(strPapers) > 0: papers = pd.concat([strPapers,papers])
        papers = papers.drop(columns=['str_node']).reset_index(drop=True)

        #'Scientist'
        scientists = nodes[nodes['nodeType']=='Scientist'].copy()
        if len(scientists) > 0:
            scientists['nodeID'] = scientists.apply(lambda x:metadataGraph._getScientist_ID(x), axis=1)
            scientists['displayName'] = scientists.apply(lambda x:x['node'].get('name'), axis=1)
            scientists['institutions'] = scientists.apply(lambda x:x['node'].get('institutions'), axis=1)
        else:
            scientists = pd.DataFrame(columns=['nodeID','displayName'])


        # 'Institutions', 
        insti = scientists[['institutions']].copy().explode('institutions').rename(columns={'institutions':
                                                                                            'node'}).dropna()
        if len(insti) > 0:
            insti['displayName'] = insti['node'].apply(lambda x:metadataGraph._extract(x,'name')) 
            insti['address'] = insti['node'].apply(lambda x:metadataGraph._extract(x,'address'))
            insti['country'] = insti['address'].apply(lambda x:metadataGraph._country(x)).replace('Russian Federation','Russia')
            insti = insti.drop_duplicates(subset=['displayName','address'])
            insti['nodeID'] = insti['displayName']+insti['address']
            
            print(insti.head(1))
            # Locations
            locations = insti[['country']].rename(columns={'country':'displayName'}).reset_index(drop=True).drop_duplicates()
            if len(locations) > 0:
                locations['nodeID'] = locations['displayName']
                locations['nodeType'] = 'Location'
            else:
                locations = pd.DataFrame(columns=['nodeID','displayName'])
            
        else:
            insti = pd.DataFrame(columns=['nodeID','displayName'])
            locations = pd.DataFrame(columns=['nodeID','displayName'])

        def get_name(x):
            try:
                return x.get('name')
            except:
                print(x)
                return x.get('name')
        # 'Conference', 'Journal',  'Book'
        venues = nodes[nodes['nodeType'].isin(['Conference','Journal','Book'])].copy()
        if len(venues) > 0:
            print(venues.shape)
            print(venues.head(1))
            venues['displayName'] = venues.apply(lambda x:get_name(x['node']), axis=1) 
            venues['nodeID'] = venues['displayName']
            venues = venues.drop_duplicates(subset=['displayName']).reset_index(drop=True) 
        else:
            venues = pd.DataFrame(columns=['nodeID','displayName'])
            

        # 'Topic/Expertise',
        topics = nodes[nodes['nodeType']=='Topic/Expertise'].copy() 
        if len(topics) > 0:
            topics['nodeID'] = topics.apply(lambda x:x['node'].get('name'), axis=1)
            topics['displayName'] = topics['nodeID']
            topics['tag_type'] = topics.apply(lambda x:x['node'].get('tag_type'), axis=1)
            topics = topics[~topics['tag_type'].isin(['Multidisciplinary','SUBJABBR','manufacturer','tradename'])].copy()
        else:
            topics = pd.DataFrame(columns=['nodeID','displayName'])

        # 'Funding Agency', 
        fundingAgencies = nodes[nodes['nodeType']=='Funding Agency'].copy().drop_duplicates().reset_index(drop=True)
        if len(topics) > 0:
            fundingAgencies['displayName'] = fundingAgencies['node'] 
            fundingAgencies['nodeID'] = fundingAgencies['displayName']
            fundingAgencies = fundingAgencies.drop_duplicates(subset=['displayName'])
        else:
            topics = pd.DataFrame(columns=['nodeID','displayName'])


        nodesDataframe = pd.concat([papers, scientists, insti, locations, venues, topics, fundingAgencies]) 
        nodesDataframe = nodesDataframe.reset_index(drop=True)[['node','nodeID','nodeType','displayName']].copy()
        
        nodesDataframe = nodesDataframe[nodesDataframe['nodeID']!='REMOVE ME'].copy()
                
        return nodesDataframe
        
    def _linksDataframe(ldf, nodesMapping):
        
        ### create link dataframes 
        
        
        ldf = ldf[~ldf['from'].astype(str).isin(NodeRepresentationsToDrop)].copy()
        ldf = ldf[~ldf['to'].astype(str).isin(NodeRepresentationsToDrop)].copy()
        
        # scientist-institution links
        
        nodes = pd.concat([ldf[['from','fromNodeType']].copy().rename(columns={'from':'node','fromNodeType':'nodeType'}),
                           ldf[['to','toNodeType']].copy().rename(columns={'to':'node','toNodeType':'nodeType'})])
        nodes['nodeType'] = nodes.apply(metadataGraph._sourceType,axis=1)
 
        #'Scientist'
        scientists = nodes[nodes['nodeType']=='Scientist'].copy()
        if len(scientists) > 0:
            scientists['nodeID'] = scientists.apply(lambda x:metadataGraph._getScientist_ID(x), axis=1)
            scientists['displayName'] = scientists.apply(lambda x:x['node'].get('name'), axis=1)
            scientists['institutions'] = scientists.apply(lambda x:x['node'].get('institutions'), axis=1)
        else:
            scientists = pd.DataFrame(columns=['nodeID','displayName'])
        
        scilinks = scientists[['nodeID','nodeType','institutions']].copy().rename(columns={'nodeID':'from',
                                                                                    'nodeType':'fromNodeType'})
        scilinks = scilinks.explode('institutions')
        scilinks = scilinks.dropna(subset=['institutions']).reset_index()
        scilinks['to'] = scilinks['institutions'].apply(lambda x:x.get('name'))
        scilinks['toNodeType'] = 'Institution'
        scilinks['edgeType'] = 'Belongs To'
        scilinks = scilinks.drop(columns=['institutions','index'])
        scilinks
 
        nodesWithMapping = nodesMapping.keys()

        linksForNodes = ldf[(ldf['from'].astype(str).isin(nodesWithMapping)) & 
                            (ldf['to'].astype(str).isin(nodesWithMapping))].copy()
        linksForNodes['from'] = linksForNodes['from'].apply(lambda x:nodesMapping[str(x)])
        linksForNodes['to'] = linksForNodes['to'].apply(lambda x:nodesMapping[str(x)])

        fundingAgencies = linksForNodes[linksForNodes['toNodeType']=='Funding Agency'].copy()
        fundingAgencies = fundingAgencies.rename(columns={'from':'to','fromNodeType':'toNodeType',
                                                         'to':'from','toNodeType':'fromNodeType'})
        fundingAgencies['edgeType'] = fundingAgencies['edgeType'].replace('Funded By','Funded')
        linksForNodes = linksForNodes[linksForNodes['toNodeType']!='Funding Agency'].copy()


        linksDataframe = pd.concat([scilinks,fundingAgencies,linksForNodes]).reset_index(drop=True)
        linksDataframe['edgeType'] = linksDataframe['edgeType'].replace('Authored','Author Of')
        
        
        # remove nodes without IDs  (nodeID='REMOVE ME')
        linksDataframe = linksDataframe[linksDataframe['from']!='REMOVE ME'].copy()
        linksDataframe = linksDataframe[linksDataframe['to']!='REMOVE ME'].copy()
        
        
        return linksDataframe
        
        
    def restructure_links(data):
        """
        Format extracted links data into the nodes dataframe and links dataframe. 
        returns nodesDF, linksDF
        """         
        if type(data) is str:
            data = pd.read_json(data, lines=True)
            
        # load data and explode lists of nodes in from/to columns as needed 
        data['from'] = data['from'].apply(metadataGraph._listify)
        data['to'] = data['to'].apply(metadataGraph._listify)
        data = data.explode('from')
        data = data.explode('to')
        data = data.dropna(subset=['from','to']).reset_index()
        
        # Nodes DataFrame
        nodesDataframe = metadataGraph._nodesDataframe(data)
        nodesMapping = {str(nodeStr):nodeID for nodeStr, nodeID in zip(nodesDataframe['node'], nodesDataframe['nodeID'])}
        
         
        ### create link dataframes 
        linksDataframe = metadataGraph._linksDataframe(data, nodesMapping)

        return nodesDataframe, linksDataframe

    
    def restructure_links_by_chunks(loadfp, chunksize, 
                                    saveNodesFN=None, saveLinksFN=None):
        """
        Format extracted links data into the nodes dataframe and links dataframe,
        iterating through a large file by chunks. 
        saves to files instead of returning
        """  
        print(loadfp, ' starting...', time.ctime())
        if saveNodesFN is None:
            saveNodesFN = loadfp.replace('.json','--Nodes-Dataframe.json')
        if saveLinksFN is None:
            saveLinksFN = loadfp.replace('.json','--Nodes-Dataframe.json')
            
        
        nodesMapping = {}
        for ldf in pd.read_json(loadfp, lines=True, chunksize=chunksize):
            #
            nodesDataframe = metadataGraph._nodesDataframe(ldf)
            
            # save nodes dataframe to file
            with open(saveNodesFN,'a') as f:
                f.write(nodesDataframe.to_json(orient='records',lines=True)+'\n')
                
            for nodeStr, nodeID in zip(nodesDataframe['node'], nodesDataframe['nodeID']):
                nodesMapping[str(nodeStr)] = nodeID 
        print(saveNodesFN, ' complete!', time.ctime())
       
        for ldf in pd.read_json(loadfp, lines=True, chunksize=chunksize):
            
            linksDataframe = metadataGraph._linksDataframe(ldf, nodesMapping)
            
            # save nodes dataframe to file
            with open(saveLinksFN,'a') as f:
                f.write(linksDataframe.to_json(orient='records',lines=True)+'\n')
        
        print(saveLinksFN, ' complete!', time.ctime())
        
    
    
import glob
import datetime
    
def loadNodes(datasource, nodeTypes=None, datadir='./', verbose=True): 
    if verbose: print(datetime.datetime.now())
    if nodeTypes is None:
        node_files = [x for x in glob.glob(os.path.join(datadir,f'*{datasource}*Nodes*')) if 'Paper_summary' not in x]
    else:
        node_files = [y for x in [glob.glob(os.path.join(datadir,f'*{datasource}*Nodes*{ntype}.jsonl')) 
                                  for ntype in nodeTypes] for y in x]
        
    print('Node Files:')
    for i in node_files: print('\t'+i)
    
    print(datetime.datetime.now())
    print('\nLoading nodes...')
    nodes = []
    for node_file in node_files:
        dfi = pd.read_json(node_file,lines=True).dropna(subset=['nodeID'])
        if len(dfi) > 0:
            dfi['nodeID'] = dfi['nodeID'].apply(lambda x:x.strip())
            nodes.append(dfi)
    nodes = pd.concat(nodes,sort=False).drop(columns=['node','date','institutions',#'address','country',
                                           'origin','doi',
                                           #'title','abstract',
                                           'name'],errors='ignore')
    print(nodes.shape)
    print(pd.DataFrame(nodes.nodeType.value_counts()))
    return nodes


def loadData(datasource, datadir='./'): 
    print(datetime.datetime.now())
    node_files = [x for x in glob.glob(os.path.join(datadir,f'*{datasource}*Nodes*')) if 'Paper_summary' not in x]
    edge_files = [x for x in glob.glob(os.path.join(datadir,f'*{datasource}*Links*')) if 'Paper_summary' not in x]
    print('Node Files:')
    for i in node_files: print('\t'+i)
    print('\nEdge Files:')
    for i in edge_files: print('\t'+i)
        
    print(datetime.datetime.now())
    print('\nLoading nodes...')
    nodes = []
    for node_file in node_files:
        dfi = pd.read_json(node_file,lines=True).dropna(subset=['nodeID'])
        if len(dfi) > 0:
            dfi['nodeID'] = dfi['nodeID'].apply(lambda x:x.strip())
            nodes.append(dfi)
    nodes = pd.concat(nodes,sort=False).drop(columns=['node','date','institutions', 
                                           'origin','doi', 
                                           'name'],errors='ignore') 
    print(nodes.shape)
    print(pd.DataFrame(nodes.nodeType.value_counts()))
    print(datetime.datetime.now())
    print('\nLoading edges...')
    edges = []
    for edge_file in edge_files: 
        edges.append(pd.read_json(edge_file,lines=True))
    edges = pd.concat(edges,sort=False).dropna(subset=['from','to'])
    edges['from'] = edges['from'].apply(lambda x:x.strip())
    edges['to'] = edges['to'].apply(lambda x:x.strip())
    print(edges.shape)
    print(pd.DataFrame(edges.edgeType.value_counts()))
    return nodes, edges    
    
def loadDataSubset(datasource, nodetypes, edgetypes, datadir='./'): 
    print(datetime.datetime.now())
    node_files = []
    edge_files = []
    for typeX in nodetypes:
        node_files.extend(glob.glob(os.path.join(datadir,f'*{datasource}*Nodes*{typeX}.jsonl'.replace(' ','*'))))
    for typeX in edgetypes:
        edge_files.extend(glob.glob(os.path.join(datadir,f'*{datasource}*Links*{typeX}.jsonl'.replace(' ','*'))))
    print('Node Files:')
    for i in node_files: print('\t'+i)
    print('\nEdge Files:')
    for i in edge_files: print('\t'+i)
       
    if len(node_files)>0:
        print(datetime.datetime.now())
        print('\nLoading nodes...')
        nodes = []
        for node_file in node_files:
            if 'Paper_summary' in node_file: continue
            dfi = pd.read_json(node_file,lines=True) 
            if len(dfi) > 0:
                dfi['nodeID'] = dfi['nodeID'].apply(lambda x:x.strip())
                nodes.append(dfi)
        if len(nodes) > 0:
            nodes = pd.concat(nodes).drop(columns=['node','date', 
                                                   'origin','doi', 
                                                   'name'],errors='ignore')
            nodes = nodes[nodes['nodeType']!='Topic'].copy()
            print(nodes.shape)
            print(pd.DataFrame(nodes.nodeType.value_counts()))
        else:
            nodes = pd.DataFrame()
    else:
        nodes = pd.DataFrame()
    if len(edge_files)>0:
        print(datetime.datetime.now())
        print('\nLoading edges...')
        edges = []
        for edge_file in edge_files: 
            edges.append(pd.read_json(edge_file,lines=True))
        if len(edges) > 0:
            edges = pd.concat(edges)
            edges['from'] = edges['from'].apply(lambda x:x.strip())
            edges['to'] = edges['to'].apply(lambda x:x.strip())
            print(edges.shape)
            print(pd.DataFrame(edges.edgeType.value_counts()))
        else:
            edges = pd.DataFrame()
    else:
        edges = pd.DataFrame() 
    return nodes, edges
    





