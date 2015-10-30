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

filename = "Classifiers/classifier.pkl"
classifier2 = joblib.load(filename)
vectorizer = joblib.load('Classifiers/vectorizer.pkl')

@app.route('/api/v1.0/classify', methods=['POST'])
def classify():
	print("Classifier request")

	json = request.json
	print(json)
	test = json['text']
	print(test)
	test_vector = vectorizer.transform([test])
	#print(test_vector)
	#print(test_vector.toarray())
	result = classifier2.predict(test_vector)
	print(result)
	return result[0]

if __name__ == '__main__':
	print("Started flask server")
	app.run(debug=True)