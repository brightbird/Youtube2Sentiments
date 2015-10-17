__author__ = 'ytay2'

'''
Script to build Features
Current feature set implementation
1) Document -> Word2vec average vectors 

'''

from sklearn import svm
# import libraries, modules
import parser
import gensim, logging
import os 
from gensim.models import Word2Vec
from nltk import word_tokenize
import nltk
import csv
import re
import numpy as np
import sys
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk import bigrams
import pprint
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score



"Build word vector averages from Word2vec"
def buildWordVector(model, text, size):
	vec = np.zeros(size).reshape((1, size))
	count = 0.
	for word in text:
		try:
			vec += model[word].reshape((1, size))
			count += 1.
		except KeyError:
			continue
	if count != 0:
		vec /= count
	#print(vec)
	return vec[0]

def runSVM():
	#Perform classification with SVM, kernel=rbf
	classifier_rbf = svm.SVC()
	t0 = time.time()
	classifier_rbf.fit(train_vectors, train_labels)
	t1 = time.time()
	prediction_rbf = classifier_rbf.predict(test_vectors)
	t2 = time.time()
	time_rbf_train = t1-t0
	time_rbf_predict = t2-t1

	# Perform classification with SVM, kernel=linear
	classifier_linear = svm.SVC(kernel='linear')
	t0 = time.time()
	classifier_linear.fit(train_vectors, train_labels)
	t1 = time.time()
	prediction_linear = classifier_linear.predict(test_vectors)
	t2 = time.time()
	time_linear_train = t1-t0
	time_linear_predict = t2-t1

	# Perform classification with SVM, kernel=linear
	classifier_liblinear = svm.LinearSVC()
	t0 = time.time()
	classifier_liblinear.fit(train_vectors, train_labels)
	t1 = time.time()
	prediction_liblinear = classifier_liblinear.predict(test_vectors)
	t2 = time.time()
	time_liblinear_train = t1-t0
	time_liblinear_predict = t2-t1

	# Print results in a nice table
	print("Results for SVC(kernel=rbf)")
	print("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
	print(classification_report(test_labels, prediction_rbf))
	print("Accuracy:", accuracy_score(test_labels,prediction_rbf))
	print("Results for SVC(kernel=linear)")
	print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
	print(classification_report(test_labels, prediction_linear))
	print("Accuracy:", accuracy_score(test_labels,prediction_linear))
	print("Results for LinearSVC()")
	print("Training time: %fs; Prediction time: %fs" % (time_liblinear_train, time_liblinear_predict))
	print(classification_report(test_labels, prediction_liblinear))
	print("Accuracy:", accuracy_score(test_labels,prediction_liblinear))

#main script execution
if __name__ == "__main__":
	#load model
	model = gensim.models.Word2Vec.load('models/model_music')
	
	print("Building feature sets...")
	patternForSymbol = re.compile(r'(\ufeff)', re.U)
	train_data =[]
	train_labels=[]
	test_labels=[]
	test_data=[]
	train_vectors = []
	test_vectors = []
	size = 400 #feature size for word2vec model
	print("Reading dataset..")
	#reads in CSV file
	with open('data/dataset.csv','rb') as dataFile:
		reader = csv.reader(dataFile, delimiter=',')
		for index,row in enumerate(reader):
			row[0] = row[0].decode('utf-8')
			rowEdited = re.sub(patternForSymbol, '', row[0])
			comment = rowEdited if rowEdited != "" else row[0]
			sentiment = row[1]
			comment = comment.lower()
			#convert comment to word vectors
			words = word_tokenize(comment)   
			vector = buildWordVector(model,words,size)
			if(index>320):
				test_data.append(comment)
				test_vectors.append(vector)
				test_labels.append(sentiment)
				continue
			else:
				train_data.append(comment)
				train_vectors.append(vector)
				train_labels.append(sentiment)
	#print(len(train_vectors[0][0]))

	classifier_rbf = svm.SVC()
	t0 = time.time()
	classifier_rbf.fit(train_vectors, train_labels)
	t1 = time.time()
	prediction_rbf = classifier_rbf.predict(test_vectors)
	t2 = time.time()
	time_rbf_train = t1-t0
	time_rbf_predict = t2-t1

	# Perform classification with SVM, kernel=linear
	classifier_linear = svm.SVC(kernel='linear')
	t0 = time.time()
	classifier_linear.fit(train_vectors, train_labels)
	t1 = time.time()
	prediction_linear = classifier_linear.predict(test_vectors)
	t2 = time.time()
	time_linear_train = t1-t0
	time_linear_predict = t2-t1

	# Perform classification with SVM, kernel=linear
	classifier_liblinear = svm.LinearSVC()
	t0 = time.time()
	classifier_liblinear.fit(train_vectors, train_labels)
	t1 = time.time()
	prediction_liblinear = classifier_liblinear.predict(test_vectors)
	t2 = time.time()
	time_liblinear_train = t1-t0
	time_liblinear_predict = t2-t1

	# Print results in a nice table
	print("Results for SVC(kernel=rbf)")
	print("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
	print(classification_report(test_labels, prediction_rbf))
	print("Accuracy:", accuracy_score(test_labels,prediction_rbf))
	print("Results for SVC(kernel=linear)")
	print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
	print(classification_report(test_labels, prediction_linear))
	print("Accuracy:", accuracy_score(test_labels,prediction_linear))
	print("Results for LinearSVC()")
	print("Training time: %fs; Prediction time: %fs" % (time_liblinear_train, time_liblinear_predict))
	print(classification_report(test_labels, prediction_liblinear))
	print("Accuracy:", accuracy_score(test_labels,prediction_liblinear))