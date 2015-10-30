'''
Script for
Classification with sklearn feature extraction
TF IDF TfidfTransformer + SVM classification
'''


import sys
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from nltk import word_tokenize
import nltk
import csv
import re
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk import bigrams
import pprint
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score


#main script execution
if __name__ == "__main__":

	print("Running SVM Module...")
	patternForSymbol = re.compile(r'(\ufeff)', re.U)
	train_data =[]
	train_labels=[]
	test_labels=[]
	test_data=[]

	#reads in CSV file
	with open('Dataset/dataset.csv','r') as dataFile:
		reader = csv.reader(dataFile, delimiter=',')
		for index,row in enumerate(reader):
			row[0] = row[0].decode('utf-8')
			rowEdited = re.sub(patternForSymbol, '', row[0])
			comment = rowEdited if rowEdited != "" else row[0]
			sentiment = row[1]
			if(index>320):
				test_data.append(comment)
				test_labels.append(sentiment)
				continue
			else:
				train_data.append(comment)
				train_labels.append(sentiment)

	vectorizer = TfidfVectorizer(min_df=5,
								 max_df = 0.8,
								 sublinear_tf=True,
								 use_idf=True)
	train_vectors = vectorizer.fit_transform(train_data)
	test_vectors = vectorizer.transform(test_data)

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
	