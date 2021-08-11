import xml.etree.ElementTree as ET
import glob
from lxml import etree
import json
import xmltodict
import pprint
import tqdm

import glob
import json
import pprint
import pandas as pd

from iteration_utilities import unique_everseen

class GrobidParser:

    def __init__(self,fn=None,xml_dir=None,json_dir='./',verbose=True):

        self.fn = fn
        self.path = xml_dir
        self.json_dir = json_dir
        self.verbose = verbose

        if self.fn is None and self.path is None:
            self.json_files = glob.glob(json_dir)
        
    def to_json(self):
    
        if not self.fn is None:
            print(f'Parsing {self.fn}')
            json_output = self.parse_xml(self.fn)
            outfile = self.json_dir + '/' + self.fn.replace('.xml','.json').split('/')[-1]
            self.json_files = [outfile]
            print(f'Saving to {outfile}')
            with open(outfile,'w') as f:
                json.dump(json_output,f)
        elif not self.path is None:
            print(f'Parsing files in {self.path} and saving to {self.json_dir}')
            self.json_files = []
            for fn in tqdm.tqdm(glob.glob(self.path)):
                json_output = self.parse_xml(fn)
                outfile = self.json_dir + '/' + fn.replace('.xml','.json').split('/')[-1]
                self.json_files.append(outfile)
                with open(outfile,'w') as f:
                    json.dump(json_output,f)

    def authors_from_json(self):

        print(self.json_files)
        print(f'Extracting authors from {len(self.json_files)} files')
        authors = []
        for fn in tqdm.tqdm(self.json_files):
            with open(fn,'r') as f:
                paper = json.load(f)

            author_subset = []
            for a, auth in enumerate(paper['authors']):
                try:
                    parsed = self.parse_author(auth)
                    if parsed is None:
                        continue
                except:
                    print(f'Failed to parse author from {fn}')
                    print(auth)
                    print('------'*20)
                    
                author_subset.append(parsed)


            for a,parsed in enumerate(author_subset):
                
                parsed['paper'] = fn.split('/')[-1].split('.')[0]
                parsed['author_order'] = a
                if a == len(author_subset) - 1:
                    parsed['last_author'] = True
                else:
                    parsed['last_author'] = False
                if a == 0:
                    parsed['first_author'] = True
                else:
                    parsed['first_author'] = False
                authors.append(parsed)
                
        return pd.DataFrame(authors)

    def parse_author(self,author):
    
        auth = {}

        if 'persName' in author.keys():

            name = author['persName']

            if 'forename' in name.keys():

                first = name['forename']

                if type(first) is dict:
                    auth[first['@type']] = first['#text']
                else:
                    for n in first:
                        auth[n['@type']] = n['#text']


            if 'surname' in name.keys():
                auth['surname'] = name['surname']


            if 'email' in author.keys():
                auth['email'] = author['email']

            if 'affiliation' in author.keys():

                if type(author['affiliation']) is dict:
                    affs = [author['affiliation']]
                else:
                    affs = author['affiliation']

                for aff in affs:
                    del aff['@key']


                affs = list(unique_everseen(affs))
                a = 0
                for aff in affs:

                    if a == 5:
                        break

                    if 'orgName' in aff.keys():

                        if 'address' in aff.keys():
                            add = aff['address']

                            if 'settlement' in add.keys():
                                auth[f'org_city_{a+1}'] = add['settlement']

                            if 'country' in add.keys():
                                if type(add['country']) is dict:
                                    auth[f'org_country_{a+1}'] = add['country']['#text']
                                    auth[f'org_countrycode_{a+1}'] = add['country']['@key']
                                else:
                                    auth[f'org_country_{a+1}'] = add['country']


                        if type(aff['orgName']) is dict:
                            orgs = [aff['orgName']]
                        else:
                            orgs = aff['orgName']

                        for org in orgs:
                            org_type = org['@type']

                            if '@key' in org.keys():
                                org_type = org['@key']

                            auth[f"org_{org_type}_{a+1}"] = org['#text']
                        a += 1


            return(auth)
                
    def parse_xml(self,fn):

        tree = etree.parse(fn)
        root = tree.getroot()
        
        parsed = {}

        authors = []
        url = '{http://www.tei-c.org/ns/1.0}'
        p = ['teiHeader','fileDesc','sourceDesc','biblStruct','analytic','author']
        p = '/'.join([url + x for x in p])
        for author in root.findall(p):
            auth = xmltodict.parse(etree.tostring(author))
            auth = json.loads(json.dumps(auth))['author']

            to_del = []
            for k in auth.keys():
                if '@' in k:
                    to_del.append(k)
            for k in to_del:
                del auth[k]

            authors.append(auth)

        parsed['authors'] = authors

        for title in root.findall('{http://www.tei-c.org/ns/1.0}teiHeader/{http://www.tei-c.org/ns/1.0}fileDesc/{http://www.tei-c.org/ns/1.0}titleStmt')[0]:
            parsed['title'] = etree.tostring(title, encoding='utf8', method='text').decode("utf-8")


        for abstract in root.findall('{http://www.tei-c.org/ns/1.0}teiHeader/{http://www.tei-c.org/ns/1.0}profileDesc')[0]:
            parsed['abstract'] = etree.tostring(abstract, encoding='utf8', method='text').decode("utf-8")

        parsed['content'] = []
        for child in root.findall('{http://www.tei-c.org/ns/1.0}text/{http://www.tei-c.org/ns/1.0}body/{http://www.tei-c.org/ns/1.0}div'):
            content_dict = {}
            for div in child:

                content = etree.tostring(div, encoding='utf8', method='text').decode("utf-8")
                if 'head' in div.tag:
                    content_dict['title'] = content
                    content_dict['text'] = ''
                else:
                    if 'text' in content_dict.keys():
                        content_dict['text'] += content
                    else:
                        content_dict['text'] = content

            parsed['content'].append(content_dict)

        return(parsed)
