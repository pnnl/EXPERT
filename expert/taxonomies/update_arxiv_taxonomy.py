import pandas as pd
import requests
from bs4 import BeautifulSoup
import json


if __name__ == "__main__":
    url = 'https://arxiv.org/category_taxonomy'
    r = requests.get(url)
    text = r.text
    soup = BeautifulSoup(text, features="lxml")
    
    links = soup.find_all('h4')

    mapping = {}
    for link in links:
        if '<span>' in str(link): 
            k,v = link.getText().replace(')','').split(' (')
            mapping[k] = v
            
    with open('arxiv_taxonomy.py', 'w') as f:
        f.write("arxiv_taxonomy = "+json.dumps(mapping, indent=4))

