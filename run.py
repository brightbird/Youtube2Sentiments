__author__ = 'ytay2'

'''
This is the final script
Creates features via word2vec and passes it to SVM for classification
Results are still terribly bad for now.
Uses word2vec average from music model trained from word2vec
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
from sklearn.feature_extraction.text import TfidfVectorizer
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



"Build word vector averages from Word2vec"
def buildWordVector(model, text, size):
	vec = np.zeros(size).reshape((1, size))
	count = 0.
	for word in text:
		word = stemmer.stem(word)
		word = lmtzr.lemmatize(word)
		try:
			vec += model[word].reshape((1, size))
			count += 1.
		except KeyError:
			continue
	if count != 0:
		vec /= count
	#print(vec)
	#print(len(vec[0]))
	return vec[0]

def runLinearSVM():
	# Perform classification with SVM, kernel=linear
	print("================Results for SVC(kernel=linear)========")
	classifier_linear = svm.SVC(kernel='linear')
	t0 = time.time()
	classifier_linear.fit(train_vectors, train_labels)
	t1 = time.time()
	prediction_linear = classifier_linear.predict(test_vectors)
	t2 = time.time()
	time_linear_train = t1-t0
	time_linear_predict = t2-t1
	print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
	print(classification_report(test_labels, prediction_linear))
	#print("Accuracy:", accuracy_score(test_labels,prediction_linear))

def runRbfSVM():
	#Perform classification with SVM, kernel=rbf
	classifier_rbf = svm.SVC()
	t0 = time.time()
	classifier_rbf.fit(train_vectors, train_labels)
	t1 = time.time()
	prediction_rbf = classifier_rbf.predict(test_vectors)
	t2 = time.time()
	time_rbf_train = t1-t0
	time_rbf_predict = t2-t1
	# Print results in a nice table
	print("================Results for SVC(kernel)-RBF========")
	print("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
	print(classification_report(test_labels, prediction_rbf))
	#print("Accuracy:", accuracy_score(test_labels,prediction_rbf))

def runLibLinearSVM():
	# Perform classification with SVM, kernel=linear
	classifier_liblinear = svm.LinearSVC()
	t0 = time.time()
	classifier_liblinear.fit(train_vectors, train_labels)
	t1 = time.time()
	prediction_liblinear = classifier_liblinear.predict(test_vectors)
	t2 = time.time()
	time_liblinear_train = t1-t0
	time_liblinear_predict = t2-t1
	print("================Results for LibLinear SVC========")
	print("Training time: %fs; Prediction time: %fs" % (time_liblinear_train, time_liblinear_predict))
	print(classification_report(test_labels, prediction_liblinear))
	#print("Accuracy:", accuracy_score(test_labels,prediction_liblinear))

#main script execution
if __name__ == "__main__":
	#load model
	#model = gensim.models.Word2Vec.load('Models/model_music')
	model = Word2Vec.load_word2vec_format('Dataset/GoogleNews-vectors-negative300.bin', binary=True)  # C binary format
	print("Building feature sets...")
	patternForSymbol = re.compile(r'(\ufeff)', re.U)
	train_data =[]
	train_labels=[]
	test_labels=[]
	test_data=[]
	train_vectors = []
	test_vectors = []
	size = 300 #feature size for word2vec model


	#Stemmer and Lemmatizer
	lmtzr = WordNetLemmatizer()
	stemmer = PorterStemmer()

	#stores vectors before spliting into training and testing
	ListOfFeatures = []

	print("Reading dataset..")
	#reads in CSV file
	with open('Dataset/dataset.csv','rb') as dataFile:
		reader = csv.reader(dataFile, delimiter=',')
		for index,row in enumerate(reader):
			if(index==0):
				continue
			row[0] = row[0].decode('utf-8')
			rowEdited = re.sub(patternForSymbol, '', row[0])
			comment = rowEdited if rowEdited != "" else row[0]
			sentiment = row[1]
			comment = comment.lower()
			words = word_tokenize(comment)   
			vector = buildWordVector(model,words,size)
			feature = {}	#Feature dictionary
			feature['word2vec'] = vector
			feature['comment'] = comment
 			feature['sentiment'] = sentiment
			ListOfFeatures.append(feature)
	totalrows = len(ListOfFeatures)
	print(len(ListOfFeatures))
	random.shuffle(ListOfFeatures)


	for index,feature in enumerate(ListOfFeatures):
		vector = feature['word2vec']
		sentiment = feature['sentiment']
		if(index>500):
			test_vectors.append(vector)
			test_labels.append(sentiment)
			continue
		else:
			train_vectors.append(vector)
			train_labels.append(sentiment)
	runLinearSVM()
	#runRbfSVM()
	#runLibLinearSVM