"""
Created on Mon Nov 29 09:14:43 2021

@author: Thijs Schoppema
@function: parse the TREC documents to a json file
"""
import unlzw3
from pathlib import Path

import gzip
import os
import json
from bs4 import BeautifulSoup

OUTPUT_DIRECTORY = "./json_files"
INPUT_DIRECTORY = "../data"
GZIP_EXTENSION = ('.gz', '.0z', '.1z', '.2z')
REDO = False

def read_dirs(in_dir):
    """Reads from a directory recusively all sub-files and 
    writes them over to a json file after the data has been processed

    Args:
        in_dir (str): The relative path to the target dir
    """    
    if not REDO:
        with open("errors.txt", "w+") as f:
            f.write("Errors:\n")
    
    for doc in os.scandir(in_dir):
        if os.path.isdir(doc):
            read_dirs(doc)
        elif doc.name.lower().endswith(GZIP_EXTENSION) and "read" not in doc.name.lower():
            doc_name = os.path.splitext(doc.name)[0]
            output_path = os.path.join(OUTPUT_DIRECTORY, doc_name+'.json')
            if os.path.isfile(output_path) and not REDO: continue
            
            print('Decompressing', doc.path, 'to', output_path)
            if doc.name.lower().endswith(GZIP_EXTENSION[0]):
                with gzip.open(doc.path, 'r') as doc:
                    html_doc = doc.read()
            else: html_doc = unlzw3.unlzw(Path(doc.path))
                
            json_doc, errors = parse_doc(BeautifulSoup(html_doc, 'html.parser'))
                
            if errors: 
                with open("errors.txt", "a") as f:
                    errors = 'FILE NAME: ' + doc_name + '\n\n' + errors + '\n\n'
                    f.write(errors)
                
            with open(output_path, 'w+') as output_file:
                json.dump(json_doc, output_file, indent=4, sort_keys=True)

def parse_doc(docs):
    """Formats a HTML document to JSON document for later applications

    Args:
        docs (Object): Parsed HTML doc by BeautifulSoup

    Returns:
        dictionary: The JSON document in a dictionary
        errors: errors found while parsing the file
    """    
    json_list = []
    n = 0
    e = 0
    doc_errors = []
    for doc in docs.find_all("doc"):

        json_file = {"id":None, "header":None, "contents":None}
        docid = doc.find("docno")
        headline = doc.find("headline")
        text = doc.find("text")
        
        if docid and text:
            json_file["id"] = docid.string.strip()
            if headline: json_file["header"] = headline.get_text().strip() #json.dumps(headline.get_text().strip())
            if text:
                json_file["contents"] =  text.get_text().strip() #json.dumps(text.get_text().strip())
                json_list.append(json_file)

        else: 
            doc_errors.append(docid.string.strip())
            e += 1
        n += 1
    if e != 0:
        errors = '\ntotal docs processed: ' + str(n)
        errors += '\ntotal errors: ' + str(e)
        errors += "\ndoc id's: " + ', '.join(doc_errors)
        errors += "\n"+'-'*30+"\n"
        
    else: errors = None
    return json_list, errors

read_dirs(INPUT_DIRECTORY)