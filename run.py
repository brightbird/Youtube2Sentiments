__author__ = 'ytay2'

'''
Run Script for NLP modules
Currently supports:
1) Feature selection (Bag Of words, TFIDF, Word2Vec averaging, Word2Vec similarity scores)
2) Training and testing classifiers (SVM Linear only)
3) K-fold validation with support with equal token validation
4) Switches for Pos Vs NEG or relevant vs Irrelevant
5) Switches for Preprocessing (lem or stemming)
'''

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
from settings import * 


"Build word vector averages from Word2vec"
def buildWordVector(model, text, size):
	global vectorCount
	global key_error_rate
	global entireTextFailed
	vectorCount+=1
	errorCount = 0
	count = 0
	vec = np.zeros(size).reshape((1, size))
	count = 0.
	for word in text:
		if(STEMMING):
			word = stemmer.stem(word)
		if(LEMMATIZE):
			word = lmtzr.lemmatize(word)
		try:
			#print(model[word].shape)
			vec += model[word].reshape((1, size))
			count += 1
		except KeyError:
			errorCount += 1
			continue
	if count != 0:
		vec /= count
		errorRate = (errorCount/(count+errorCount)) * 100
		#print("Error Percentage:"+str(errorRate))
		key_error_rate += errorRate 
	
	if count == 0:
		key_error_rate += 100
		entireTextFailed += 1
		#print("Entire text failed")
	#print(vec.shape)
	return vec[0]

"Training & Testing method for SVM "
#Pass Mode in so we know which vectorizer the classifier belongs to
#Classifier and Vectorizer comes in a pair!
def runLinearSVM(mode):
	global SAVED
	global saveVectorizer
	# Perform classification with SVM, kernel=linear
	print("================Results for SVC(kernel=linear)========")
	classifier_linear = svm.SVC(kernel='linear')
	t0 = time.time()
	classifier_linear.fit(train_vectors, train_labels)
	t1 = time.time()
	prediction_linear = classifier_linear.predict(test_vectors)
	t2 = time.time()
	time_linear_train = t1 - t0
	time_linear_predict = t2 - t1
	print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
	print(classification_report(test_labels, prediction_linear))
	score = accuracy_score(test_labels, prediction_linear)
	print(score)
	# print("Accuracy:", accuracy_score(test_labels,prediction_linear))
	print("\n")
	if (score >= SAVE_THRESHOLD and SAVED == False and mode!='COMBINE' and mode!='W2V'):
		#Mode not equal COMBINE because saving 2 vectorizers is troublesome for API usage! :(
		#Save Classifier for future use
		print("Saving classifier of score:" + str(score))
		saveClassifier(classifier_linear,mode)
		SAVED = True  # switch, do not want to save multiple classifiers
		saveVectorizer = True  # switch, yes we want to persist vectorizer
	return score


"Save Classifier to file"
def saveClassifier(classifier,mode):
	if(RELEVANCY):
		filename = 'Classifiers/classifier_relevancy_'+mode+'.pkl'
	else:
		filename = 'Classifiers/classifier_posneg_'+mode+'.pkl'
	joblib.dump(classifier, filename, compress=9)
	print("Classifier persisted!")


def createClassifierInput(dataSets, validationSet = 0):
	global index, feature, vector, sentiment, comment, positiveCount, negativeCount
	training1 = dataSets[(validationSet + 1) % 3]
	training2 = dataSets[(validationSet + 2) % 3]
	partition = len(training1) + len(training2) -1
	ListOfFeatures = training1 + training2 + dataSets[validationSet]
	for index, feature in enumerate(ListOfFeatures):
		if (WORD2VEC): vector = feature['word2vec']
		sentiment = feature['sentiment']
		if(sentiment=='positive'):
				positiveCount+=1
		elif(sentiment=='negative'):
				negativeCount+=1
		comment = feature['comment']  # raw text for TF-IDF vectorization
		if (index > (partition)):
			test_data.append(comment)
			if (WORD2VEC):
				test_vectors.append(vector)
			test_labels.append(sentiment)
			continue
		else:
			train_data.append(comment)
			if (WORD2VEC):
				train_vectors.append(vector)
			train_labels.append(sentiment)


# main script execution
if __name__ == "__main__":
	#load model
	if(WORD2VEC):
		print("Loading Model..may take some time..please wait!")
		model = gensim.models.Word2Vec.load('Models/model'+str(size))
		#model = Word2Vec.load_word2vec_format('Dataset/GoogleNews-vectors-negative300.bin', binary=True)  # C binary format
	print("Building feature sets...")


	print("Reading dataset..")
	#reads in CSV file
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
			
			#Switch to check if we want to do Pos/Neg or Relevant/Irrelant
			if(RELEVANCY):
				#Do Relevant / Irrelevant Classify
				if(sentiment=="irrelevant"):
					irrelevantCount+=1
				else:
					relevantCount+=1
					sentiment = 'relevant'
			else:
				if(sentiment=='positive'):
					positiveCount+=1
				elif(sentiment=='negative'):
					negativeCount+=1
				elif(sentiment=='neutral'):
					neutralCount+=1
					if(POSNEG):continue #We only want PosNeg comparisons
				elif(sentiment=='irrelevant'):
					irrelevantCount+=1
					if(POSNEG):continue
				elif(sentiment=='mixed'):
					mixedCount+=1
					if(POSNEG):continue

			comment = comment.lower()
			words = word_tokenize(comment)
			words = [w for w in words if not w in stopwords.words("english")]
			#print(words)
			if(POSFilters):
				tags = nltk.pos_tag(words)
				filteredWords = []
				for tag in tags:
					if(tag[1] in POSFilter):
						filteredWords.append(tag[0])
				print(filteredWords)
				words = filteredWords

			#Creating features
			feature = {}	#feature dictionary for filtering later

			#Building Vectors
			if(WORD2VEC):
				vector = buildWordVector(model,words,size)
				feature['word2vec'] = vector

			feature['comment'] = comment	#save raw text for future processing
			feature['sentiment'] = sentiment
			ListOfFeatures.append(feature)

	totalrows = len(ListOfFeatures)
	partition = (totalrows) * TRAINING_PERCENTAGE




	dataSets = []
	#random.shuffle(ListOfFeatures)
	if(equalTokens):
		dataSets = equalDataSetSplitter.splitIntoThreeEqualTokenSet(ListOfFeatures)
	for set in dataSets:
		equalDataSetSplitter.tokenChecker(set)
	#Main loop
	for i in range(0,iteration):
		random.shuffle(ListOfFeatures)
		train_data =[]
		train_labels=[]
		test_labels=[]
		test_data=[]
		train_vectors = []
		test_vectors = []


		"Note:Please use a switch to control new functionality, so I do not lose old functionality"
		if(equalTokens):
			# Constructing actual input for classifier
			# Based on number of tokens
			createClassifierInput(dataSets, i%3)
		else:
			#Constructing actual input for classifier
			for index,feature in enumerate(ListOfFeatures):
				if(WORD2VEC):vector = feature['word2vec']
				sentiment = feature['sentiment']

				comment = feature['comment'] #raw text for TF-IDF vectorization
				if(index>(partition)):
					test_data.append(comment)
					if(WORD2VEC):
						test_vectors.append(vector)
					test_labels.append(sentiment)
					continue
				else:
					train_data.append(comment)
					if(WORD2VEC):
						train_vectors.append(vector)
					train_labels.append(sentiment)

		#Runs word2 vec first because we do not want to corrupt the word2vec vector variable
		if(WORD2VEC):
			#Running once for Word2Vec approach (Averaging)
			print("----------Word2vec Approach------------")
			score = runLinearSVM("W2V")
			WORD2VEC_MAX+=score

		if(COMBINE):
			print("----------TF-IDF + Word2Vec---------")
			temp_train_vectors = TFIDFvectorizer.fit_transform(train_data)
			temp_test_vectors = TFIDFvectorizer.transform(test_data)
			#print(temp_train_vectors.shape)
			#print(train_vectors[0].shape)
			combined_train_vector = []
			combined_test_vector = []
			for index,vector in enumerate(train_vectors):
				#print(temp_train_vectors[index,:].shape)
				#print(vector.shape)
				temp = sparse.hstack((vector,temp_train_vectors[index,:]))
				#print("tempshape:" + str(temp.shape))
				temp = temp.toarray()[0]
				#print(temp.shape)
				combined_train_vector.append(temp)
			for index,vector in enumerate(test_vectors):
				temp = sparse.hstack((vector,temp_test_vectors[index,:]))
				#print(temp)
				temp = temp.toarray()[0]
				#print(temp.shape)
				combined_test_vector.append(temp)
			train_vectors = combined_train_vector
			test_vectors = combined_test_vector
			#print(train_vectors[0].shape)
			score = runLinearSVM("COMBINE")
			COMBINE_MAX+=score

		if(TFIDF):
			print("----------Tf-idf Approach------------")
			train_vectors = TFIDFvectorizer.fit_transform(train_data)
			test_vectors = TFIDFvectorizer.transform(test_data)
			score = runLinearSVM("TFIDF")
			TFIDF_MAX+=score
			if(saveVectorizer):
				print("Saving TF-IDF vectorizer..")
				if(RELEVANCY):
					filename = 'Classifiers/vectorizer_relevancy_TFIDF.pkl'
				else:
					filename = 'Classifiers/vectorizer_posneg_TFIDF.pkl'
				joblib.dump(TFIDFvectorizer, filename, compress=9)
				saveVectorizer = False

		if(BOW):
			print("----------Bag of words--------------")
			train_vectors = BOWvectorizer.fit_transform(train_data)
			#vocab = BOWvectorizer.get_feature_names()
			#print(vocab)
			test_vectors = BOWvectorizer.transform(test_data)
			score = runLinearSVM("BOW")
			BOW_MAX+=score
			if(saveVectorizer):
				print("Saving BOW vectorizer..")
				if(RELEVANCY):
					filename = 'Classifiers/vectorizer_relevancy_BOW.pkl'
				else:
					filename = 'Classifiers/vectorizer_posneg_BOW.pkl'
				joblib.dump(BOWvectorizer, filename, compress=9)
				saveVectorizer = False

		if (W2W_SIM_SENTIMENT):
			print("----------- Word2Vec similarity--------------")
			train_vectors = featureExtractorW2V.getFeatures(train_data, model)
			test_vectors = featureExtractorW2V.getFeatures(test_data, model)
			featureExtractorW2V.testContext(train_data, model)
			score = runLinearSVM("W2VSIM")
			W2W_SIM_SENTIMENT_MAX += score
			print("Score: ", score)
			# print("Persisting SVM...")
			# joblib.dump(classifier_linear, 'svm_linear.pkl')

		#print("Persisting SVM...")
		#joblib.dump(classifier_linear, 'svm_linear.pkl')

if(WORD2VEC):
	print("==================Word2Vec Model Evaluation===========")
	print("Average Key Error Rate:" + str(key_error_rate/vectorCount) + "%")
	print("Entire Document Fail Rate:" + str((entireTextFailed*100)/vectorCount) + "%")

print("================Dataset Statistics====================")
if(RELEVANCY):
	print("Running in Relevancy Mode")
	print("Relevant Count:"+str(relevantCount))
	print("Irrelevant Count:" + str(relevantCount))
elif(POSNEG):
	print("Positive Count:" + str(positiveCount))
	print("Negative Count:" + str(negativeCount))
#print("total dataset size:" + str(totalrows))
#print("training size:" + str(partition))

print("================Printing Average Results=============")
if (TFIDF): print("TDIF average score:" + str(TFIDF_MAX / iteration))
if (BOW): print("Bag of words avg score:" + str(BOW_MAX / iteration))
if (WORD2VEC): print("Word2Vec Avg score:" + str(WORD2VEC_MAX / iteration))
if (COMBINE): print("COMBINE Avg score:" + str(COMBINE_MAX / iteration))
if (W2W_SIM_SENTIMENT): print("Word2Vec SIM-ALGO score:" + str(W2W_SIM_SENTIMENT_MAX / iteration))

