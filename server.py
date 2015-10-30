__author__ = 'tayyi'

'''
Python-Flask Web Service for Hosting Classifier as an API
Web Service - Api for classifier
'''

# import libraries, modules
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
from flask import Flask, jsonify, request

app = Flask(__name__)


classifier_relevancy_TFIDF = joblib.load("Classifiers/classifier_relevancy_TFIDF.pkl")
vectorizer_relevancy_TFIDF = joblib.load("Classifiers/vectorizer_relevancy_TFIDF.pkl")
classifier_relevancy_BOW = joblib.load("Classifiers/classifier_relevancy_BOW.pkl")
vectorizer_relevancy_BOW = joblib.load("Classifiers/vectorizer_relevancy_BOW.pkl")

#Positive-Negative Classifiers
classifier_posneg_BOW = joblib.load("Classifiers/classifier_posneg_BOW.pkl")
vectorizer_posneg_BOW = joblib.load('Classifiers/vectorizer_posneg_BOW.pkl')

@app.route('/api/v1.0/classify', methods=['POST'])
def classify():
	print("Classifier request")
	json = request.json
	print(json)
	test = json['text']
	#option = json['option']
	#print("Request for "+option)

	#Do relevancy check first

	test_vector = vectorizer_relevancy_BOW.transform([test])
	result = classifier_relevancy_BOW.predict(test_vector)
	if(result[0]=='irrelevant'):
		print(result[0])
		return result[0]
	else:		
		test_vector = vectorizer_posneg_BOW.transform([test])
		result = classifier_posneg_BOW.predict(test_vector)
		print(result)
		return result[0]
		

if __name__ == '__main__':
	print("Started flask server")
	app.run(debug=True)