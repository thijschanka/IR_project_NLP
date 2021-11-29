'''
Function: convert the test querries to an usable format
Created Date: Monday, November 29th 2021, 1:35:55 pm
Author: Thijs Schoppema
'''

import gzip
import os
from bs4 import BeautifulSoup

OUTPUT_FILE = "./output/queries.txt"
INPUT_FILE = "../test_documents/04.testset.gz"

def parse_query_file(in_file, out_file, inc_title=True, inc_desc=False):
    """Read the contents of the input file and parse it to generate
    an usable querry file.

    Args:
        in_file (str): The relative path to the input file
        out_file (str): The relative path to the output file
        inc_title (bool, optional): To include the title of the topic. Defaults to True.
        inc_desc (bool, optional): To include the description of the topic. Defaults to False.
    """    
    
    with gzip.open(in_file, 'r') as doc:
        html_doc = doc.read()
    topics = BeautifulSoup(html_doc, 'html.parser')
    querries = []
    for topic in topics.find_all("top"):
        number = topic.find('num').get_text().split("Number:")[1].split("\n", 1)[0].strip()
        if inc_title:
            title = parse_component(topic.find("title"), "<desc>")
            #title = topic.find("title").get_text().split("\n\n")[0].strip()
            if title != '':
                querries.append(number + '\t' + title.strip())
        if inc_desc:
            desc = parse_component(topic.find("desc"), "<narr>")
            #desc = topic.find("desc").get_text().split("\n", 1)[1].split("\n\n")[0].strip()
            if desc != '':
                if "Description:" in desc:
                    desc = desc.split(":", 1)[1].strip()
                querries.append(number + '\t' + desc.strip())
    with open(out_file, 'w') as f:
        f.write('\n'.join(querries))

def parse_component(txt, breakpoint):
    """Read a multiline string until the breakpoint has been found

    Args:
        txt (str): A (multiline) string
        breakpoint (str): a string where the parsing should stop

    Returns:
        str: All lines of the string until the breakpoint
    """    
    lines = []
    txt = str(txt).split(">",1)[1]
    for line in txt.split('\n'):
        if breakpoint not in line:
            lines.append(line.strip())
        else: return ' '.join(lines)
    return ' '.join(lines).strip()

parse_query_file(INPUT_FILE, OUTPUT_FILE)
