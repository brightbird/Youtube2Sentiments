# Youtube2Sentiments
Youtube2Sentiments is developed for school project. It is still a work in progress.
It uses word2vec models and svm for classification of Youtube comments. We are still
experimenting/tinkering with stuff.

Yes all the corpus, dataset folders are gitignored. 

## Current development

#### Classifier Scripts
Note: Performance measured on self-obtained and annotated Youtube comment dataset.
1. **naive.py** - Uses NLTK's naive bayes or logistic regression [65%-ish]
2. **tfidf-svm.py** - Uses Sklearn TF-IDF vectorizer + SVM (linear & RBF kernals) [75%-ish] 
3. **run.py** - Experiments with different vector representations (bag of words, tfidf, and word2vec) [79-86%-ish]
	1. Bag of words - average performance: 84-ish%
	2. TFIDF - average performance: 82-ish%
	3. Word2Vec(using avg) - average performance :79%-ish (on own word2vec model, word2vec model is probably still bad)
	4. Combine(TF-IDF+Word2Vec concatenate) - average performance: 82-ish%
4. **testClassifier** -- test persisted classifier/vectorizer using CLI

#### Corpus related Scripts
1. **corpusBuilder.py** -- Parse class that does a few things
	1. Cleaning Corpus crawled from a variety of sources (Twitter, Youtube etc.)
	2. Prepare Corpus for Word2Vec training (into Sentences)

#### Word2Vec related Scripts
1. **trainWord2Vec.py** -- Train a Word2Vec, calls corpusBuilder to prepare input
2. **testWord2Vec.py** -- Test Word2Vec with self prescribed test cases
3. **bin2model.py** -- binary vectors to persisted models (this doesn't actually speed things up.)

#### Directories
1. /Models - All trained models are persisted and loaded
2. /Crawlers - Contain crawling scripts 
3. /Corpus - Prepared and cleaned Corpus (all appended to 1 txt file)
4. /Raw - All Raw text files from crawlers 
5. /Tests - Currently Empty


## Version
0.0.2