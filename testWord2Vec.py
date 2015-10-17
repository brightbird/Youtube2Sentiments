#!/usr/bin/python
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

#main script execution
if __name__ == "__main__":
	print("Choose Corpus to train from:")
	choice = raw_input("Select your data source:\n 1) Music Model \n 2) Google News Model \n 3) Quit \n")

	print("Loading models..")
	#load models
	if(choice=="1"):
		model = gensim.models.Word2Vec.load('models/model_music')
	elif(choice=="2"):
		print("Loading Google news vectors...")
		model = Word2Vec.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
	print("Model loaded")

	#Run tests
	print("Running trials and tests..")
	testcase = ['music','guitar','taylor','justin','piano','vocal','tone','acoustic','vanessa','beethoven','chopin']
	for test in testcase:
		try:
			result = model.most_similar(positive=[test], topn=10)
			print("=========RESULT========")
			print("test case: " + test)
			for result in result:
				print(result)
		except:
			print "Test word not in vocab"
