__author__ = "ytay2"

'''
Script to test persisted bag of words classifier
Performance is terrible! 
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

patternForSymbol = re.compile(r'(\ufe  ff)', re.U) #Regex Emoticon Cleaner
filename = "Classifiers/classifier.pkl"
classifier = joblib.load(filename)
vectorizer = joblib.load('Classifiers/vectorizer.pkl')
print(vectorizer)
print("loaded classifier and vectorizer")

test = raw_input("Classifier Test: \n")
test_vector = vectorizer.transform([test])
print(test_vector.toarray())
result = classifier.predict(test_vector)
print(result)

#This enumarates dataset and checks results
'''
with open('Dataset/dataset.csv','rb') as dataFile:
		reader = csv.reader(dataFile, delimiter=',')
		for index,row in enumerate(reader):
			if(index==0):
				print("Skipping header for data file")
				continue

			#Pre-processing
			row[0] = row[0].decode('utf-8')
			rowEdited = re.sub(patternForSymbol, '', row[0])
			comment = rowEdited if rowEdited != "" else row[0]
			sentiment = row[1]
			comment = comment.lower()
			print(comment)
			test = vectorizer.transform([comment])
			result = classifier.predict(test)
			print(result)
			print("-----------")
'''