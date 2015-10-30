#!/usr/bin/python
import sys

__author__ = 'ytay2'

'''
Word2Vec Module for NLP Project
Command Line Script to Train a Word2Vec model
Uses gensim (python word2vec implementation)
Supports selecting data source for training word2vec model
'''

# import libraries, modules
import parser
import gensim, logging
import os
from gensim.models import Word2Vec
from nltk import word_tokenize
import nltk
import csv
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import *
from nltk.corpus import stopwords

stem=False
lemmatize=False 

# main script execution
if __name__ == "__main__":
    print("Choose Corpus to train from:")
    choice = raw_input("Select your data source:\n 1) Domain Model \n 2) Google News Model  \n 3) Quit \n")

    print("Loading models..")
    model = None
    # load models
    if (choice == "1"):
        model = gensim.models.Word2Vec.load('Models/model28')
    elif (choice == "2"):
        print("Loading Google news vectors...")
        # model = gensim.models.Word2Vec.load('Models/model_GoogleNews_300')
        model = Word2Vec.load_word2vec_format('Dataset/GoogleNews-vectors-negative300.bin',
                                              binary=True)  # C binary format
    elif (choice == "3"):
        sys.exit(0)
    print("Model loaded")

    # Run tests
    print("Running trials and tests..")
    testcase = ['chopin', 'liszt', 'beethoven', 'justin', 'violin', 'vocal', 'girl', 'happy', 'sad', 'gay']
    for test in testcase:
        try:
            result = model.most_similar(positive=[test], topn=10)
            print("=========RESULT========")
            print("test case: " + test)
            for result in result:
                print(result)
        except:
            print ("Test word not in vocab")
	#NLTK Lemmatization and Stemming
	if (lemmatize or stem):
		lmtzr = nltk.stem.wordnet.WordNetLemmatizer()
		stemmer = PorterStemmer()
		
	print("Loading models..might take awhile")
	#load models
	model = gensim.models.Word2Vec.load('Models/model100')
	#model = Word2Vec.load_word2vec_format('Dataset/GoogleNews-vectors-negative300.bin', binary=True)  # C binary format
	print("Model loaded")
	tests = []
	#Read test file
	with open('Tests/tests.txt') as f:
		lines = f.readlines()
		for line in lines:
			line = line.decode('utf-8')
			line = line.strip()
			tests.append(line)
	#Run tests
	print("Running tests..")
	for test in tests:
		if(stem):test = stemmer.stem(test)
		if(lemmatize):test = lmtzr.lemmatize(test)
		print(test)	
		try:
			result = model.most_similar(positive=[test],topn=10)
			print("=========RESULT========")
			print("test case: " + test)
			for result in result:
				print(result)
		except:
			print "Test word not in vocab"
