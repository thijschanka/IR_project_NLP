'''
Filename: d:\master_opleiding\IR\project\IR_project_NLP\word_emb.py
Path: d:\master_opleiding\IR\project\IR_project_NLP
Created Date: Friday, December 10th 2021, 11:57:18 am
Author: Thijs Schoppema

Copyright (c) 2021 Your Company
'''

import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords

#import tensorflow as tf
#import tensorflow_hub as hub

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

import torch
# Load the relevant files
def load_data(tokenize=True):
    """A function that loads and tokenizes (if specified) the generated collection file in move_data.py.

    Args:
        tokenize (bool, optional): To tokenize the data. Defaults to True.

    Returns:
        df (Pandas DataFrame): Pandas dataframe containing the whole tokenized collection with identifiers of each document.
    """    
    df = pd.read_json("./output/json_collection/complete_collection.json", orient='records', encoding='utf8')

    if tokenize:        
        df['contents'] = df["contents"].apply(lambda x: word_tokenize(x.lower()))
        df.to_csv("./output/json_collection/complete_collection_tokenized.csv")
    return df

def load_queries():
    """A function that loads the queries

    Returns:
        queries (Dictionary): A dictionary  that contains all queries in the given file, the ID is used as the key and the question/content as the value.
    """    
    queries = dict()
    with open("./output/queries.txt", "r") as f:
        for line in f:
            cols = line.split("\t")
            queries[cols[0].strip()] = cols[1].strip()
    return queries

# Generate embeddings for online model
def generate_embeddings(df, embed_method):
    """A function that generates embeddings for a given tensorflow model

    Args:
        df (Pandas Dataframe): The dataframe containing all the documents in the collection
        embed_method (Object): A Keraslayer model retrieved from Tensorflow hub

    Returns:
        (List of vectors): An ordered list of vector where each vector represents one document in the collection
    """    
    sent = nltk.flatten(df["contents"].to_list())
    return embed_method(sent)

# Compare embeddings
def cosine_sim(embeddings_col, embeddings_query):
    """Calculates the cosine similarity between two vectors

    Args:
        embeddings_col (np array): The embedded vectors of each document in the collection
        embeddings_query (np array): The embedding of the query

    Returns:
        Float: The cosine similarity between the two vectors
    """    
    dot_product = np.sum(np.multiply(np.array(embeddings_col),np.array(embeddings_query)),axis=1)
    prod_sqrt_magnitude = np.multiply(np.sum(np.array(embeddings_col)**2,axis=1)**0.5, np.sum(np.array(embeddings_query)**2,axis=1)**0.5)
    
    cosine_sim = dot_product/prod_sqrt_magnitude
    return cosine_sim

# doc retrieval
def doc_retrieval(query, embeddings, docs, model, local_model, sim_method='cos'):
    """A function that embeds the query and returns a ranking of relevant documents based on the standardized calculated similarity

    Args:
        query (List of one string): A list containing the querry for which the documents will be retrieved
        embeddings (np array): The embedded vectors of each document in the collection
        docs (List of strings): A list of the identifiers of each document in the embeddings variable
        model (Object): The model used to generate the embeddings
        local_model (bool): If the model is a local model made with gensim or a loaded keras model from Tensorflow hub
        sim_method (str, optional): Which similarity method to use, currently only cosine similarity is supported. Defaults to 'cos'.

    Returns:
        values (List of floats): A sorted list of similarity's, sorted descending.
        ranked_docs (List of Strings): A sorted list of document identifiers, the list is sorted based on the ordering of the value variable
    """    
    if local_model:
        embeddings_query = model.infer_vector(word_tokenize(query[0].lower()))
    else:
        embeddings_query = model(query)
    if sim_method == 'cos':
        sim = cosine_sim(np.array(embeddings), np.array(embeddings_query).reshape(1, len(embeddings_query)))
    
    standardized_sim  = (sim-min(sim))/(max(sim)-min(sim))
    values, ranked_docs = zip(*sorted(zip(standardized_sim, docs), reverse=True))
    return values, ranked_docs

# Train embeddings
def train_embeddings(data, tokenize=True, max_epochs=20, vec_size=100, dm=0, mvs=None, dbow=0, input="contents", model_name='./output/results/word_emb_dm1.model'):
    
    # if dm == 0 it is bag of words else it preserves word order
    print("tokenize")
    #if tokenize:
    #    data = [TaggedDocument(words=word_tokenize(doc["contents"].lower()), tags=[doc["id"]]) for doc in data]
    #else:
    #    data = [TaggedDocument(words=doc["contents"].lower(), tags=[doc["id"]]) for doc in data]
    
    print("define model")
    model = Doc2Vec(vector_size=vec_size,
                    dm=dm ,
                    seed=2021,
                    max_vocab_size=mvs,
                    dbow_words=dbow)
    print("make vocab")
    model.build_vocab(data)
    
    print("start training")
    model.train(data,
                total_examples=model.corpus_count,
                epochs=max_epochs
                )
    model.save(model_name)
    print("Model Saved")

# Perform the task
def main(model_loc = "https://tfhub.dev/google/nnlm-en-dim128/2", n_hits = 10000, it = 'Q0', run='wrb04', local_model=False):
    print("load data")
    df = load_data(tokenize=True)
    queries = load_queries()
    print("load model")
    if local_model:
        model = Doc2Vec.load(model_loc)
        data = [doc["contents"] for ind, doc in df.iterrows()]
        print("compute embeddings")
        embeddings = [model.infer_vector(d) for d in data]
    else:
        model = hub.KerasLayer(model_loc)
        embeddings = generate_embeddings(df, model)
    
    results = []
    print("compare")
    for q_id in queries:
        print(q_id)
        values, hits = doc_retrieval([queries[q_id]], embeddings, df['id'].to_list(), model, local_model)
        for i in range(0, min(len(hits), n_hits)):
            d = hits[i]
            score = values[i]
            
            results.append([str(q_id), str(it), str(d), str(i+1), str(score), str(run)])
    output_str = "\n".join([' '.join(i) for i in results])
    open("./output/results/word_emb_dm1_vec512.run", "w").write(output_str)

RETRAIN = True
TOKENIZE = True

if RETRAIN:
    with open("./output/json_collection/complete_collection.json", encoding='utf8') as f:
        data = json.load(f)
    
    if TOKENIZE:
        print("tokenize")
        data = [TaggedDocument(words=word_tokenize(doc["contents"].lower()), tags=[doc["id"]]) for doc in data]
    else:
        data = [TaggedDocument(words=doc["contents"].lower(), tags=[doc["id"]]) for doc in data]
    
    print(torch.cuda.get_device_name(0))
    train_embeddings(data, tokenize=True, max_epochs=20, vec_size=200, dm=1, mvs=None, dbow=0, input="contents", name='./output/results/word_emb_dm1_vec200.model')
    train_embeddings(data, tokenize=True, max_epochs=20, vec_size=200, dm=0, mvs=None, dbow=0, input="contents", name='./output/results/word_emb_dm0_vec200.model')
    train_embeddings(data, tokenize=True, max_epochs=20, vec_size=100, dm=1, mvs=None, dbow=0, input="contents", name='./output/results/word_emb_dm1_vec100.model')

main(model_loc = "./output/dm_robust04_dm1_vec512.model", local_model=True)