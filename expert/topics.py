import glob
import pandas as pd
import numpy as np
import scipy.sparse as ss
from corextopic import corextopic as ct
from langdetect import detect_langs, detect
import time
import json
import pickle
import networkx as nx
import re
import string

from wordcloud import WordCloud
from wordcloud import get_single_color_func

import graphviz
from graphviz import Digraph, Graph

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer

from matplotlib import pyplot as plt
import seaborn as sns

def clean_text(text):
    text = re.sub(r'\d+', '', text)
    text = "".join([char.lower() for char in text if char not in string.punctuation]) 
    text = text.replace('et al','')
    text = re.sub('\s+', ' ', text).strip()
    return text

class SimpleGroupedColorFunc(object):
    """Create a color function object which assigns EXACT colors                                                                           
       to certain words based on the color to words mapping                                                                                
                                                                                                                                           
       Parameters                                                                                                                          
       ----------                                                                                                                          
       color_to_words : dict(str -> list(str))                                                                                             
         A dictionary that maps a color to the list of words.                                                                              
                                                                                                                                           
       default_color : str                                                                                                                 
         Color that will be assigned to a word that's not a member                                                                         
         of any value from color_to_words.                                                                                                 
    """

    def __init__(self, color_to_words, default_color):
        self.word_to_color = {word: color
                              for (color, words) in color_to_words.items()
                              for word in words}

        self.default_color = default_color

    def __call__(self, word, **kwargs):
        return self.word_to_color.get(word, self.default_color)


class GroupedColorFunc(object):
    """Create a color function object which assigns DIFFERENT SHADES of                                                                    
       specified colors to certain words based on the color to words mapping.                                                              
                                                                                                                                           
       Uses wordcloud.get_single_color_func                                                                                                
                                                                                                                                           
       Parameters                                                                                                                          
       ----------                                                                                                                          
       color_to_words : dict(str -> list(str))                                                                                             
         A dictionary that maps a color to the list of words.                                                                              
                                                                                                                                           
       default_color : str                                                                                                                 
         Color that will be assigned to a word that's not a member                                                                         
         of any value from color_to_words.                                                                                                 
    """

    def __init__(self, color_to_words, default_color):
        self.color_func_to_words = [
            (get_single_color_func(color), set(words))
            for (color, words) in color_to_words.items()]

        self.default_color_func = get_single_color_func(default_color)

    def get_color_func(self, word):
        """Returns a single_color_func associated with the word"""
        try:
            color_func = next(
                color_func for (color_func, words) in self.color_func_to_words
                if word in words)
        except StopIteration:
            color_func = self.default_color_func

        return color_func

    def __call__(self, word, **kwargs):
        return self.get_color_func(word)(word, **kwargs)



class wordcloud:

    def __init__(self, freq_dict, *, title=None, color=None, colormap=None, exact_colors=False, default_color = 'black',
                 background_color="white", width=300, height=225, prefer_horizontal=1,
                 interpolation="bilinear", fontsize=20, saveFN=None):

        if type(color) is str:
            color = {color:freq_dict.keys()}

        if colormap is not None:
            wc = WordCloud(background_color=background_color, width=width, height=height,
                           prefer_horizontal=prefer_horizontal, colormap=colormap)
        else:
            if exact_colors:
                # use single tone for each color                                                                                           
                grouped_color_func = SimpleGroupedColorFunc(color, default_color)
            else:
                # use multiple tones for each color (randomly assigned)                                                                    
                grouped_color_func = GroupedColorFunc(color, default_color)

            wc = WordCloud(background_color=background_color, width=width, height=height,
                           prefer_horizontal=prefer_horizontal, color_func=grouped_color_func)

        wc.generate_from_frequencies(freq_dict)

        # show  
        #plt.figure(figsize=8,6))
        plt.imshow(wc, interpolation=interpolation)
        if title is not None: plt.title(title, fontsize=fontsize)
        plt.axis("off")
        if saveFN is not None: plt.savefig(saveFN, dpi=300)
        plt.show()

class TopicModeler:

    def __init__(self,data,text_col='abstract',
                 max_vocab=10000,max_ngram=3, 
                 output_path = './',
                 count='binarize'):

        self.df = data
        self.count = count
        self.max_vocab=max_vocab
        self.max_ngram = max_ngram
        self.output_path = output_path

        self.text = list(self.df[text_col].apply(clean_text))
        self.tokenize()
        
        self.best_result = 0
        self.ns = []
        self.tcs = []
        self.times = []


    def tokenize(self):

        token = RegexpTokenizer(r'[a-zA-Z0-9]+')
        self.cv = CountVectorizer(lowercase=True,stop_words='english',
                                  ngram_range = (1,self.max_ngram),
                                  tokenizer = token.tokenize,
                                  max_features=self.max_vocab)
        self.text_counts = self.cv.fit_transform(self.text)
        self.words = self.cv.get_feature_names()

    def fit_model(self,n_topics=10,set_model=False):

        print(f'Fitting topic model for n={n_topics}')
        self.tm = ct.Corex(n_topics, count=self.count)
        self.tm.fit(self.text_counts,words = self.words)

        if set_model:
            self.best_model = self.tm

    def evaluate(self):
        
        return(self.tm.tcs.sum())

    def find_best_n(self,n_min=10,n_max=100,n_step=10):

  
        for n in range(n_min,n_max + 1,n_step):
            
            start = time.time()
            self.fit_model(n)
            end = time.time()
            elapsed = (end - start)/60.0
            print(f'Fit in {elapsed} minutes')

            tc_sum = self.evaluate()

            self.ns.append(n)
            self.tcs.append(tc_sum)
            self.times.append(elapsed)

            self.results = pd.DataFrame({'n':self.ns,
                                         'corr_explained':self.tcs,
                                         'time_elapsed':self.times})
            self.results = self.results.sort_values('n')

            self.plot_results()

            if tc_sum > self.best_result:
                self.best_model = self.tm
                self.best_result = tc_sum

    def plot_results(self,save_fig=False):

        plt.plot(self.results['n'],self.results['corr_explained'],marker='o')
        plt.xlabel('Number of Topics')
        plt.ylabel('Total Correlation Explained')
        if save_fig:
            print(f'Saving {self.output_path}/correlation_by_ntopics.png')
            plt.savefig(f'{self.output_path}/correlation_by_ntopics.png')
        plt.show()

    def plot_correlation(self,save_fig=False):

        plt.bar(range(self.best_model.tcs.shape[0]), self.best_model.tcs, color='#4e79a7', width=0.5)
        plt.xlabel('Topic', fontsize=16)
        plt.ylabel('Total Correlation (nats)', fontsize=16)
        if save_fig:
            print(f'Saving {self.output_path}/correlation_by_topic.png')
            plt.savefig(f'{self.output_path}/correlation_by_topic.png')
        plt.show()

    def print_topics(self):

        topics = self.best_model.get_topics()
        for topic_n,topic in enumerate(topics):
            wds,mis,_ = zip(*topic)
            topic_str = str(topic_n+1)+': '+','.join(wds)
            print(topic_str)

    
    def plot_wordclouds(self,save_fig=False):

        topics = self.best_model.get_topics()
        colors = sns.color_palette()
        colors = colors.as_hex()
        for topic in range(len(topics)):
            topic_df = pd.DataFrame(self.best_model.get_topics(topic=topic, n_words=100)).sort_values(1,ascending=False)
            topic_df = topic_df[topic_df[1] > 0]
            word_freq = topic_df.set_index(0)[1].to_dict()
    
            print(topic)
            if save_fig:
                print(f'Saving {self.output_path}/wordcloud_{topic}.png')
                wordcloud(word_freq, color=colors[topic % 10],
                          saveFN=f'{self.output_path}/wordcloud_{topic}.png')
            else:
                wordcloud(word_freq, color=colors[topic % 10])

    
    def save_results(self):
        
        #save document-topic matrix
        doc_topics = pd.DataFrame(self.best_model.p_y_given_x)
        pd.concat([self.df,doc_topics],axis=1).to_csv(f'{self.output_path}/document_topics.csv',
                                                      index=False)


        #save topic-token lists
        max_words = max((self.best_model.alpha >= 1.).sum(axis=1))
        topic_words = self.best_model.get_topics(n_words=max_words+1)
        for i,tw in enumerate(topic_words):
            tw_df = pd.DataFrame(tw)
            tw_df.columns = ['token','mutual_information']
            tw_df.to_csv(f'{self.output_path}/topic{i+1}_terms.csv',index=False)

        #save the model
        with open(f'{self.output_path}/topic_model.pkl','wb') as f:
            pickle.dump(self.best_model,f)


    def get_topic_corr(self, thresh=0.14):

        topic_corrs = pd.DataFrame(np.exp(self.best_model.log_p_y_given_x)).corr()
        topic_corrs = topic_corrs.where(np.triu(np.ones(topic_corrs.shape)).astype(np.bool))
        topic_corrs = pd.melt(topic_corrs.reset_index(),id_vars=['index'])
        topic_corrs = topic_corrs.dropna()
        topic_corrs = topic_corrs[(topic_corrs['index'] != topic_corrs['variable']) & (topic_corrs['value'] > thresh)]
        topic_corrs['weight'] = (topic_corrs['value'] - topic_corrs['value'].min()) / (topic_corrs['value'].max() - topic_corrs['value'].min())
        topic_corrs['weight'] = 3*topic_corrs['weight'] + 1
        topic_corrs['index'] = topic_corrs['index'].astype(str)
        topic_corrs['variable'] = topic_corrs['variable'].astype(str)
        topic_corrs = topic_corrs.reset_index(drop=True)
        return topic_corrs

    
    def generate_graph(self, dataframe, source_node_column_name, target_node_column_name, weight_col,
                       *, engine='sfdp', weighted=True, scale_vis_edge_weight=1.0,
                       max_edge_weight=None, edge_weight_filter=None,
                       left_right=False, upweight_by_node=False,
                       type_colorings={}):


        g = Graph('G', engine=engine)

    
        sourceN = dataframe[[source_node_column_name]]
        sourceN.columns = ['node']
        targetN = dataframe[[target_node_column_name]]
        targetN.columns = ['node']
        nodesDF = pd.concat([sourceN, targetN])
    
        nodesDF = nodesDF.drop_duplicates()


        for n in nodesDF['node']:
            g.node(str(n), shape="none", label="", 
                   image=f'{self.output_path}/wordcloud_{n}.png')

        for i in dataframe.index:
    
            if weighted:
                weight = str(dataframe.loc[i, weight_col] * scale_vis_edge_weight)
            else:
                weight = 1
            
            g.edge(str(dataframe.loc[i,source_node_column_name]),
                   str(dataframe.loc[i, target_node_column_name]),
                   penwidth = str(weight))

        return dataframe, g

    
    def topic_corr_graph(self, thresh=0.14, scale_vis_edge_weight=1.0,engine='dot'):

        topic_corrs = self.get_topic_corr(thresh=thresh)

        _,G = self.generate_graph(topic_corrs,
                             'index','variable','weight',
                             engine=engine,
                             scale_vis_edge_weight=scale_vis_edge_weight)

        print(f'Saving {self.output_path}/topic_cooccur_graph')
        G.render(filename=f'{self.output_path}/topic_cooccur_graph')
