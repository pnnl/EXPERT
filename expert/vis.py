"""
Visualization-related code

Requires graphviz installation:

conda install -c anaconda python-graphviz

"""


import os
import graphviz
from graphviz import Digraph
import pandas as pd
from IPython.core.display import display, HTML
import re


def get_weighted_dataframe(dataframe, 
                           source_node_column_name, target_node_column_name, edge_label_column_name,
                           edge_weight_col = 'edge_weight', upweight_by_node=False, 
                           edge_weight_filter=None):

    dataframe[edge_weight_col] = 1
    dataframe = dataframe.groupby([source_node_column_name,
                              target_node_column_name,
                              edge_label_column_name],as_index=False)[edge_weight_col].sum()

    dataframe['weight'] = 1
    source_counts = dataframe.groupby(source_node_column_name)['weight'].count().reset_index()
    target_counts = dataframe.groupby(target_node_column_name)['weight'].count().reset_index()
    total_counts = source_counts.merge(target_counts,left_on=source_node_column_name,
                                       right_on=target_node_column_name,
                                       suffixes = ('_source','_target'),
                                       how='outer').fillna(0.0)
    total_counts['node_weight'] = total_counts['weight_source'] + total_counts['weight_target']

    source_weights = total_counts[[source_node_column_name,'node_weight']]
    source_weights.columns = [source_node_column_name,'source_weight']

    target_weights = total_counts[[target_node_column_name,'node_weight']]
    target_weights.columns = [target_node_column_name,'target_weight']


    dataframe = dataframe.merge(source_weights,on=source_node_column_name,how='left')
    dataframe = dataframe.merge(target_weights,on=target_node_column_name,how='left')

    if upweight_by_node:
        dataframe['weight'] = dataframe[edge_weight_col] + dataframe['source_weight'] + dataframe['target_weight']
    else:
        dataframe['weight'] = dataframe[edge_weight_col]

    if not edge_weight_filter is None:
        dataframe = dataframe[dataframe['weight'] > edge_weight_filter]

    return dataframe


def generate_metadata_graph(dataframe, source_node_column_name, target_node_column_name, edge_label_column_name,
                            *, engine='sfdp', weighted=True, scale_vis_edge_weight=1.0, 
                            edge_weight_col='edge_weight',
                            max_edge_weight=None, edge_weight_filter=None,
                            left_right=False, upweight_by_node=False, 
                            type_colorings={}):
    """
 
    :param dataframe: pandas dataframe that contains the
    relationships to include in the directed graph.
    
    :param source_node_column_name: (string) column name in
    the dataframe that includes the source node in the edge per row
    
    :param target_node_column_name: (string) column name in
    the dataframe that includes the target node in the edge per row
    
    :param edge_label_column_name: (string) column name in
    the dataframe that includes the label for the edge per row
    
    :param engine: (string) specifies the engine to use for graphviz,
    default is 'sfdp' but can also use 'neato','dot','circo'

    :param weighted: (boolean) flag that indicates whether to collapse edges and represent frequency with
    edge thickness
    
    :param scale_vis_edge_weight: (float) scale factor to apply to edge weight thickness,
     by default = 1.0 to use raw frequency as thickness
     
     
    :param type_colorings: dictionary of colors by node/edge type, 
    e.g. {'Paper':'gray', 'Scientist':'blue', 'Funded By':'green'}
    

    :return: pandas dataframe of edges, graphviz DiGraph
    """
    
    if weighted:
        dataframe = get_weighted_dataframe(dataframe, 
                                           source_node_column_name, 
                                           target_node_column_name, 
                                           edge_label_column_name,
                                           edge_weight_col=edge_weight_col,
                                           upweight_by_node=upweight_by_node, edge_weight_filter=edge_weight_filter)

    if left_right:
        g = Digraph('G', engine=engine,graph_attr={'rankdir':'LR','size' : "100" } )
    else:
        g = Digraph('G', engine=engine)
        
    if type_colorings is not None: 
        srcTransform = {source_node_column_name:'node', source_node_column_name+'NodeType':'type'}
        sourceN = dataframe[[source_node_column_name,
                             source_node_column_name+'NodeType']].rename(columns=srcTransform)
        targetTransform = {target_node_column_name:'node', target_node_column_name+'NodeType':'type'}
        targetN = dataframe[[target_node_column_name,
                             target_node_column_name+'NodeType']].rename(columns=targetTransform)
        nodesDF = pd.concat([sourceN, targetN]) 
        nodesDF = nodesDF.drop_duplicates()
        
        for n, t in zip(nodesDF['node'], nodesDF['type']):
            color = type_colorings.get(t) 
            if color is None: 
                g.node(str(n), shape='box', label=t)
            else:
                g.node(str(n), color=color, style='filled', label=t) 
        
    for i in dataframe.index: 
        edgeType = dataframe.loc[i, edge_label_column_name]
        if weighted:
            weight = str(dataframe.loc[i, 'weight'] * scale_vis_edge_weight)
        else:
            weight = 1
        if type_colorings != {}:
            edgeColor = type_colorings.get(edgeType) 
            
            if edgeColor is not None:  
                g.edge(str(dataframe.loc[i,source_node_column_name]),
                       str(dataframe.loc[i, target_node_column_name]),
                       label=str(dataframe.loc[i, edge_label_column_name]),
                       penwidth = str(weight),
                       color = edgeColor) 
            else:
                g.edge(str(dataframe.loc[i,source_node_column_name]),
                       str(dataframe.loc[i, target_node_column_name]),
                       label=str(dataframe.loc[i, edge_label_column_name]),
                       penwidth = str(weight))

    return dataframe, g


def generate_graph(dataframe, source_node_column_name,
                   target_node_column_name, edge_label_column_name,
                   *, engine='sfdp', weighted=True, scale_vis_edge_weight=1.0, 
                   edge_weight_col='edge_weight',
                   max_edge_weight=None, color_edge_over_max='red',edge_weight_filter=None,
                   left_right=False,upweight_by_node=False,collapseBidirectional=False):
    """

    e.g. for generating a graph that encodes ARG0 --Verb--> ARG1:
    source_node_column_name = 'ARG0'
    target_node_column_name = 'ARG1'
    edge_label_column_name = 'Verb'

    :param dataframe: pandas dataframe that contains the
    relationships to include in the directed graph.
    :param source_node_column_name: (string) column name in
    the dataframe that includes the source node in the edge per row
    :param target_node_column_name: (string) column name in
    the dataframe that includes the target node in the edge per row
    :param edge_label_column_name: (string) column name in
    the dataframe that includes the label for the edge per row
    :param engine: (string) specifies the engine to use for graphviz,
    default is 'sfdp' but can also use 'neato','dot','circo'

    :param weighted: (boolean) flag that indicates whether to collapse edges and represent frequency with
    edge thickness
    :param scale_vis_edge_weight: (float) scale factor to apply to edge weight thickness,
     by default = 1.0 to use raw frequency as thickness
     
     
    :param collapseBidirectional: (boolean) if true, represent edges from a-->b and b-->a as a single, bidirectional edge a<-->b

    :return: pandas dataframe of edges, graphviz DiGraph
    """

    if weighted:
        dataframe = get_weighted_dataframe(dataframe,
                                           source_node_column_name,
                                           target_node_column_name,
                                           edge_label_column_name,
                                           edge_weight_col=edge_weight_col,
                                           upweight_by_node=upweight_by_node, edge_weight_filter=edge_weight_filter)
    graph_attr = {}
    if collapseBidirectional: 
        # represent edges from a-->b and b-->a as a single, bidirectional edge a<-->b
        graph_attr['concentrate']='true'

    if left_right:
        graph_attr['rankdir']='LR'
        
    if graph_attr == {}:
        g = Digraph('G', engine=engine) 
    else:
        g = Digraph('G', engine=engine,graph_attr=graph_attr) 
        
    for i in dataframe.index:
        if weighted:
            if max_edge_weight is not None and dataframe.loc[i, 'weight'] > max_edge_weight:
                
                g.edge(dataframe.loc[i,source_node_column_name],
                       dataframe.loc[i, target_node_column_name],
                       label=dataframe.loc[i, edge_label_column_name],
                       penwidth = str(max_edge_weight * scale_vis_edge_weight),
                       color = color_edge_over_max) 
            else:
                g.edge(dataframe.loc[i,source_node_column_name],
                       dataframe.loc[i, target_node_column_name],
                       label=dataframe.loc[i, edge_label_column_name],
                       penwidth = str(dataframe.loc[i, 'weight'] * scale_vis_edge_weight))
            #
        else:
            g.edge(dataframe.loc[i,source_node_column_name],
                   dataframe.loc[i, target_node_column_name],
                   label=dataframe.loc[i, edge_label_column_name])

    return dataframe, g



def visualize(graphviz_graph, use_source=False):
    """
    Save graph visualization to pdf
    :param graphviz_graph: graphviz object for the casual graph
    """
    # display graphviz visualization of causal graph
    if use_source: graphviz_graph = graphviz.Source(graphviz_graph)
    display(graphviz_graph)


def save_to_pdf(graphviz_graph, graph_fname, *, save_dir='./', use_source=False, remove_dot_file=True):
    """
    Save graph visualization to pdf
    :param graphviz_graph: graphviz object for the casual graph
    :param save_dir: directory to save pdf image and gml file for causal graph to
    :param graph_fname: filename to use when saving to pdf e.g "narrative_graph" will
    save the graph to "narrative_graph.pdf"
    """

    # pretty sure this forces it to use 'dot' layout
    if use_source: graphviz_graph = graphviz.Source(graphviz_graph)

    # check that save_dir directory exists, otherwise create it
    os.makedirs(save_dir, exist_ok=True)

    # write to pdf
    filename=os.path.join(save_dir, f"{graph_fname}")
    graphviz_graph.render(filename=filename)
    # remove dot file
    if remove_dot_file: os.remove(filename)



    
from wordcloud import WordCloud  
from wordcloud import get_single_color_func
import matplotlib.pyplot as plt
 
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
                 background_color="white", width=300, height=200, prefer_horizontal=1,
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
        plt.imshow(wc, interpolation=interpolation)
        if title is not None: plt.title(title, fontsize=fontsize)
        plt.axis("off")
        if saveFN is not None: plt.savefig(saveFN, dpi=300)
        plt.show()



def inverse_highlight(text, keyword):
    """
    Highlights keyword occurence(s) by adding a gray background to everything else.
    """
    text = "\033[47m" + text.replace(keyword, "\033[49m" + keyword + "\033[47m")
    return text

def html_keywords_highlighted(keyword_to_color_map, text, *,
                              ignore_case=True, strong=True, background=True):
    """
    Returns HTML representation of text where each occurence of each keyword in
    keyword_to_color_map is highlighted using the color specified in keyword_to_color_map.
    E.g.
    if keyword_to_color_map = {'python':'blue'},
        all occurences of 'python' in the text are shown against
            - a blue background (if background=True) or
            - in a blue font    (if background=False)

    ignore_case is a boolean flag for whether to match using specified case of each
    character or to match the substring, regardless of case.
    E.g., if ignore_case is True,
                'python' will match to 'python', 'PYTHON', 'PyThOn'
          if ignore_case is False,
                'python' will match 'python' but not 'PYTHON' or 'PyThOn'
    """

    def colorText(x, color, strong=True):
        if strong:
            return "<strong><span style='color:" + color + "'>" + x + "</span></strong>"
        else:
            return "<span style='color:" + color + "'>" + x + "</span>"

    def colorBackground(x, color, strong=True):
        if strong:
            return "<strong><span style='background-color:" + color + ";opacity:0.8;'>" + x + "</span></strong>"
        else:
            return "<span style='background-color:" + color + ";opacity:0.8;'>" + x + "</span>"

    def highlight(x, color, strong=True):
        if background:
            return colorBackground(x, color, strong=strong)
        else:
            return colorText(x, color, strong=strong)

    for keyword in keyword_to_color_map:
        if ignore_case:
            regex = re.compile(r"{}".format(keyword), re.I)
        else:
            regex = re.compile(r"{}".format(keyword))
        output = ""
        end_of_previous_segment = 0
        for m in regex.finditer(text):
            connecting_segment = text[end_of_previous_segment:m.start()]
            matched_segment = text[m.start():m.end()]
            output += "".join([connecting_segment,
                               highlight(matched_segment, keyword_to_color_map[keyword], strong=strong)])

            end_of_previous_segment = m.end()
        output = "".join([output, text[end_of_previous_segment:]])
        text = output
    html_str = f'<html>{text}</html>'

    return HTML(html_str)

class keywordVis:
    """
    Class to wrap keyword based visualizations of text content.
    (e.g. if repeatedly using the same keyword mapping for various text analyses)
    """

    def __init__(self, keyword_to_color_map):
        self.keyword_to_color_map = keyword_to_color_map

    def show(self, text, ignore_case=True, strong=True, background=True):
        if type(text) is str:
            htmldata = html_keywords_highlighted(self.keyword_to_color_map, text,
                                                 ignore_case=ignore_case,
                                                 strong=strong, background=background)
            display(htmldata)
        elif type(text) is list:
            for t in text:
                self.show(t, ignore_case=ignore_case, strong=strong, background=background)
                display(HTML('<html><br><hr><html>'))
        elif type(text) is dict:
            for t in text:
                display(HTML('<html><strong>' + t + ':' + '</strong><html>'))
                self.show(text[t], ignore_case=ignore_case, strong=strong, background=background)

    def colorText(self, text, ignore_case=True, strong=True):
        self.show(text, ignore_case=ignore_case, strong=strong, background=False)

    def highlight(self, text, ignore_case=True, strong=True):
        self.show(text, ignore_case=ignore_case, strong=strong, background=True)


