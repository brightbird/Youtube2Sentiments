# Youtube2Sentiments
Youtube2Sentiments is developed for school project. It is still a work in progress.
It uses word2vec models and svm for classification of Youtube comments. We are still
experimenting/tinkering with stuff.


## Current development

### Experiment Results

1. Word2Vec Averages - 81.99%
2. Bag of Words (Ngrams 1-5, 5000 features) - 88.43%
3. TF-IDF (Ngrams 1-5, 851 features) - 86.20%
4. W2V Similarity Algorithm	(20 features) - 80.12%
5. TF-IDF + W2V - 91.22%

### Version 0.0.3 Updates
1. Added Flask Server to serve API
2. Word2Vec Similarity-vector Algorithm (by Espen)
3. Relevancy vs Irrelevant classifier
4. CorpusBuilder upgraded


#### Directories
1. /Models - All trained models are persisted and loaded
2. /Crawlers - Contain crawling scripts 
3. /Corpus - Prepared and cleaned Corpus (all appended to 1 txt file)
4. /Raw - All Raw text files from crawlers 
5. /Tests - Currently Empty


## Version
0.0.3