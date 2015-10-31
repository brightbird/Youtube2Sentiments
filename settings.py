# import libraries, modules
# unused imports still left because lazy 
import os
import nltk
import csv
import re
import numpy as np
import sys
import time
import math
import random
import parser
import pprint
import gensim, logging
import featureExtractorW2V
import equalDataSetSplitter
from gensim.models import Word2Vec
from nltk import word_tokenize
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk import bigrams
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import *
from sklearn.externals import joblib
from scipy import sparse


#Preprocessing Switches
LEMMATIZE = False
STEMMING = False

#Feature Selection Switches
TFIDF = True
WORD2VEC = True
BOW = True
COMBINE = True 		#TF-IDF + Word2VecAverage Combine Vectors
W2W_SIM_SENTIMENT = True	#Word2Vec Similairty Algorithm 


'''
Feature Selection
Vectorizers
'''

#Bag of words count vectorizer
BOWvectorizer = CountVectorizer(analyzer = "word",   \
							 tokenizer = None,    \
							 preprocessor = None, \
							 stop_words = None,   \
							 ngram_range=(1,5), \
							 max_features = 5000) 

# Making tf-idf vectors
TFIDFvectorizer = TfidfVectorizer(min_df=5,
								  max_df=0.8,
								  ngram_range=(1,5),
								  sublinear_tf=True,
								  use_idf=True)

#Persistence Switches
SAVED = True #True = Do not save (already saved, in other words)
saveVectorizer = False 	#Whether we want to save the vectorizer
SAVE_THRESHOLD = 0.85


#Modes 
POSNEG = True #Only evaluate Positive and Negative Polarity
RELEVANCY = False #Train for relevancy -> build relevancy classifier 

#Pre-processing Variables
patternForSymbol = re.compile(r'(\ufeff)', re.U)  # Regex Emoticon Cleaner
lmtzr = WordNetLemmatizer()
stemmer = PorterStemmer()


#POS tags Filters
POSFilter = ["JJ", "JJS", "JJR",'NN','NNS','NNP']
POSFilters = False


#Loop & Testing Controls
iteration = 10
equalTokens = False		#whether to use equal tokens partitioning 
TRAINING_PERCENTAGE = 0.9		#training and testing set ratio

#Word2Vec settings
size = 100 #feature size for word2vec model
key_error_rate = 0 #Number of key lookup errors
vectorCount = 0 	#For averaging word vectors (calculate percentage lost)
entireTextFailed = 0 	#Count for number of total failed docs

train_data =[]
train_labels=[]
test_labels=[]
test_data=[]
train_vectors = []
test_vectors = []

positiveCount=0
negativeCount=0
neutralCount=0
irrelevantCount=0
relevantCount=0 #Only for relevant vs irrelevant
mixedCount=0

#stores vectors before spliting into training and testing
ListOfFeatures = []

#Max scores for average results
TFIDF_MAX = 0
BOW_MAX = 0
WORD2VEC_MAX = 0
COMBINE_MAX = 0
W2W_SIM_SENTIMENT_MAX = 0