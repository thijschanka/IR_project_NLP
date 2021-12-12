'''
Filename: d:\master_opleiding\IR\project\IR_project_NLP\word_emb.py
Path: d:\master_opleiding\IR\project\IR_project_NLP
Created Date: Friday, December 10th 2021, 11:57:18 am
Author: Thijs Schoppema

Copyright (c) 2021 Your Company
'''

import pandas as pd
import numpy as np

import nltk
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords



def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens

#!pip install spacy
#!python -m spacy download en_core_web_sm
#!python -m spacy download en_core_web_trf

import spacy
import en_core_web_trf, en_core_web_sm

import tensorflow as tf
import tensorflow_hub as hub

def load_data(cleaned=True, accuracy=True):
    df = pd.read_json("./output/json_collection/complete_collection.json", orient='records', encoding='utf8')
        
    if cleaned:
        if accuracy:
            nlp = en_core_web_trf.load(disable=["parser", "ner"])
        else:
            nlp = en_core_web_sm.load(disable=["parser", "ner"])
        nlp.max_length = 2000000
        
        df['cleaned'] = df["contents"].apply(lambda x: prune_tokens(x, nlp))
        #df["text_sent"] = df["cleaned"].progress_apply(lambda x:re.split('\n',x))
    df["text_sent"] = df["contents"]
    return df

def load_queries():
    queries = dict()
    with open("./output/queries.txt", "r") as f:
        for line in f:
            cols = line.split("\t")
            queries[cols[0].strip()] = cols[1].strip()
    return queries

def prune_tokens(sent, nlp):
    if sent:
        tokens = []
        for word in nlp(str(sent)):
            if word.is_stop: continue
            if word.is_punct: continue
            if word.is_space: continue
            if word.like_url: continue
            if word.like_email: continue
            if word.is_currency: continue
            if word.pos_ == 'VBZ': continue
            if word.pos_ == 'ADP': continue
            if word.pos_ == 'PRON': continue
            if word.pos_ == 'AUX': continue
            
            tokens.append(word.lemma_)
    else: return np.nan
    return tokens

def generate_embeddings(df, embed_method):
    sent = nltk.flatten(df["text_sent"].to_list())
    return embed_method(sent)

def cosine_sim(embeddings_col, embeddings_query):
    dot_product = np.sum(np.multiply(np.array(embeddings_col),np.array(embeddings_query)),axis=1)
    prod_sqrt_magnitude = np.multiply(np.sum(np.array(embeddings_col)**2,axis=1)**0.5, np.sum(np.array(embeddings_query)**2,axis=1)**0.5)
    
    cosine_sim = dot_product/prod_sqrt_magnitude
    return cosine_sim

def eucl_dist(embeddings_col, embeddings_query):
    return 0

def doc_retrieval(query, embeddings, docs, embed_method, sim_method='cos'):
    recommend_docs = []
    embeddings_query = embed_method(query)
    
    if sim_method == 'cos':
        sim = cosine_sim(embeddings,embeddings_query)
    
    standardized_sim  = (sim-min(sim))/(max(sim)-min(sim))
    
    #ranked_docs = dict(zip(docs, standardized_sim))
    #ranked_docs = list(map(lambda x, y:(x,y), standardized_sim, docs))
    #ranked_docs = ranked_docs.sort(key=lambda tup: tup[0], reverse=True)
    values, ranked_docs = zip(*sorted(zip(standardized_sim, docs)))
    return values, ranked_docs

def main(module_url = "https://tfhub.dev/google/nnlm-en-dim128/2", n_hits = 100, it = 'Q0', run='wrb04'):
    df = load_data(cleaned=False, accuracy=True)
    queries = load_queries()
    
    embed = hub.KerasLayer(module_url)
    embeddings = generate_embeddings(df, embed)
    
    results = []
    for q_id in queries:
        values, hits = doc_retrieval([queries[q_id]], embeddings, df['id'].to_list(), embed)
        for i in range(0, min(len(hits), n_hits)):
            d = hits[i]
            score = values[i]
            
            results.append([str(q_id), str(it), str(d), str(i+1), str(score), str(run)])
    
    output_str = "\n".join([' '.join(i) for i in results])
    open("./output/results/word_emb.run", "w").write(output_str)

main()