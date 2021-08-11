import pandas as pd
import requests
from bs4 import BeautifulSoup
import json


if __name__ == "__main__":
    url = 'https://cran.r-project.org/web/classifications/MSC.html'
    r = requests.get(url)
    text = r.text
    soup = BeautifulSoup(text, features="lxml")
    li = soup.find_all('li')
    
    
    mapping = {}
    for l in li:
        if ':' in l.getText():
            lines = l.getText().split('\n')
            for line in lines:
                vals = line.split(':')
                code = vals[0].strip()
                label = vals[1].strip().split('(')[0]
                if label in ['None of the above, but in this section']: continue
                if code in ['91E99']: continue
                mapping[code] = label

            
    with open('msc_taxonomy.py', 'w') as f:
        f.write("msc_taxonomy = "+json.dumps(mapping, indent=4))
