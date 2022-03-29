# EXPERT

This package contains the code and notebooks developed on the EXPERT project to fuse a variety of multilingual heterogenous open-source data streams, e.g., publications, institutional web pages, conference pages, and researcher profiles, to convert unstructured data into knowledge summaries and construct dynamically evolving proliferation expertise graphs for descriptive, predictive, and prescriptive analytics.



## Install and Use

This package can be installed locally as a static install using:

``python setup.py install``

or alternatively as a development install (incorporating updates when changes are made to files) using:

``python setup.py develop``



If installed as a static install, the package must be reinstalled to incorporate updates.


If installed as a development install, changes to the files will be updated dynamically in the package without running subsequent installation commands.


### External Install Depencies
#### SPERT Model

To use the SPERT model for the local content graph:

1. Clone the SPERT repo: https://github.com/lavis-nlp/spert
2. Download the data and models:
```
bash ./scripts/fetch_datasets.sh
bash ./scripts/fetch_models.sh
```
3. Point the local graph construction to the location of the SPERT repo:
```
graph = lcg.localContentGraph(text=text,
                              spert_path='/path/to/spert/')
```


#### PDF parsing

Install GROBID using instructions here:
[https://grobid.readthedocs.io/en/latest/Install-Grobid/](https://grobid.readthedocs.io/en/latest/Install-Grobid/).

Run GROBID:
```
java -jar grobid-core/build/libs/grobid-core-0.6.2-onejar.jar -gH grobid-home -dIn /path/to/pdf/files/ -dOut /path/to/output/directory/  -exe processFullText -ignoreAssets
```

Examples for parsing the resulting XML files can be found in the GROBID Parsing Example notebook.


<hr>



## Documentation of Repository

### examples
    - Directory containing example notebooks, README in this directory summarizes examples included and functionality of each notebook.

### expert
    - Directory containing the package classes, scripts, and other code-related files for the **expert** package

#### embeddings.py  
    - code to extract ESTEEM embeddings

#### entity_resolution 
   ├── dedup.py 

    - helper functions for deduplication of entity dataframes


   ├── entityMerger.py                     
   
    - class to merge entities based on text and graph similarity


   ├── interactiveMerger.py                
   
    - wrapper class for interactive merging using the IdentityMatcher widget and entityMerger objects, maintaining history and provenance of merge choices.

 
####  global_content_graph.py

    - code to generate global content graphs from scientific publication data

####  graph_resolution.py

    - code to resolve multiple representations into single nodes in graphs

####  local_content_graph.py

    - code to generate a local content graph from scientific publication data


####  metadata

   ├── graph.py

    - code to generate a context graph from publication metadata


   ├── publications.py

    - code to format and load Scopus, Web of Science, OSTI, Arxiv, Biorxiv, and DBLP publication metadata

#### queries.py
    - nuclear related keywords and terms compiled using IAEA glossary and SME knowledge

#### srl_graph.py

    - code to generate Semantic Role Label (SRL)-based content graphs


#### taxonomies
    - mappings and scripts to update mappings of abbreviations used in arxiv, MSC, and PACS taxonomies

####  topics.py
    - topic modelling, including visualizations of related topics using wordcloud-nodes in a graph visualization
    
####  vis.py
    - visualization functions, e.g. graphviz-based graph visualizations
 
Graph Benchmarks
Graph datasets can be accessed from the Berkeley Data Cloud (BDC): https://bdc.lbl.gov/wiki/6053a11a7428ae62f36b71a5/622f96a05735d3a51055c1cb_files/
An account is required for access: https://bdc.lbl.gov/register/
The BDC is a data sharing and management platform for US Government-supported nuclear nonproliferation research projects.
        

_______________________________________________________________________________


This material was prepared as an account of work sponsored by an agency of the United States Government.  Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights.
Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.

<p align="center">
PACIFIC NORTHWEST NATIONAL LABORATORY<br/>
<i>operated by<br/>
BATTELLE<br/>
<i>for the<br/>
UNITED STATES DEPARTMENT OF ENERGY<br/>
<i>under Contract DE-AC05-76RL01830
</p>

