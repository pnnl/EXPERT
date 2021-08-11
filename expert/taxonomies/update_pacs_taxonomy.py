import pandas as pd
import requests
from bs4 import BeautifulSoup
import json


if __name__ == "__main__":
    url =  'https://ufn.ru/en/pacs/'
    r = requests.get(url)
    text = r.text
    soup = BeautifulSoup(text, features="lxml")
    li = soup.find_all('li')
    
    mapping = {}
    for t in li:
        if 'span class="pacs_num"' in str(t):
            lines = t.getText().split('\n')
            for line in lines:
                vals = line.split(' ')
                code = vals[0].strip()
                label = ' '.join(vals[1:]).strip().split('(')[0]
                if label in ['None of the above, but in this section']: continue
                if code in ['91E99']: continue
                mapping[code] = label
            
    with open('pacs_taxonomy.py', 'w') as f:
        f.write("pacs_taxonomy = "+json.dumps(mapping, indent=4))
