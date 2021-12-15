# IR_project_NLP

## Word embeddings
Convert documents and querries to embeddings, in a bag of words model
* doc2vec
* embeddings based on existing model --> Done

Compare documents and querries:
* Cosine similarity --> Done
* Dot product
* Jaccard Similarity

Pre-processing:
* remove frequent words and punctation
* Distinction between header and content of document
* Seperation of sentences

Score calculation
* Max between sentences or header and content
* First sentence of header
* Sum or Average of sentences or header and content

## Contextual embeddings
Document retrieval with BERT
* Rerankers/sparse representations --> our solution
** BM25 ranking
** BERT reranking
*** over passage monoBERT, BERT-FirstP/MaxP/SumP
*** T5, sequence2sequence model
* Dense retrieval

Pre-processing:
* remove frequent words and punctation
* Distinction between header and content of document
* Seperation of sentences
* Indexing + BM-25 ranking

Score calculation
* Max between sentences or header and content
* First sentence of header
* Sum or Average of sentences or header and content