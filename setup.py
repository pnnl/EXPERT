import os
from setuptools import setup

setup(
        name = "expert",
        version = "0.0.1",
        author = "Maria Glenski",
        author_email = "maria.glenski@pnnl.gov",
        description = ("Tools for working with EXPERT datasets"),
        packages = ["expert"],
        install_requires= [
            'elasticsearch==7.9.1',
            'pandas==0.25',
            'graphviz>=0.14.2',
            'allennlp==2.5.0',
            'spacy==3.0.0',
            'networkx',
            'scikit-learn',
            'matplotlib',
            'seaborn',
            'python-Levenshtein==0.12.0',
            'wordcloud',
            'corextopic',
            'top2vec',
            'joblib',
            'langdetect',
            'torch',
            'numpy<1.20',
            'tqdm>4.36',
            'unidecode',
            'plotly', 'chart-studio', 'cufflinks',
            'top2vec[sentence_encoders]', 'tensorflow','tensorflow_hub','tensorflow_text'
            ],
)



