#!/usr/bin/python
import sys
__author__ = 'ytay2'

'''
Word2Vec Module for NLP Project
Command Line Script to Train a Word2Vec model
Uses gensim (python word2vec implementation)
Test data from / Test directory
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
import random

stem=False
lemmatize=False 

# main script execution
if __name__ == "__main__":
		
	print("Loading models..might take awhile")
	#load models
	model = gensim.models.Word2Vec.load('Models/model28')
	#model = Word2Vec.load_word2vec_format('Dataset/GoogleNews-vectors-negative300.bin', binary=True)  # C binary format
	print("Model loaded")
	tests = []
	positive = []
	negative = []
	#Read test file
	with open('Tests/positive.txt') as f:
		lines = f.readlines()
		for line in lines:
			line = line.decode('utf-8')
			line = line.strip()
			positive.append(line)
	with open('Tests/negative.txt') as f:
		lines = f.readlines()
		for line in lines:
			line = line.decode('utf-8')
			line = line.strip()
			negative.append(line)
	#Run tests
	print("Running tests..")
	#negative = []

	for index,p in enumerate(positive):
		if(len(negative)>0):
			for n in negative:
				print(p + "-" + n)	
				try:
					result = model.most_similar(positive=[p],negative=[n],topn=10)
					print("=========RESULT========")
					for result in result:
						print(result)
				except:
					print "Test word not in vocab"
		
		else:
	
			try:
				if(index!=len(positive)):
					p2 = positive[index+1]
				else:
					p2 = positive[0]
				print(p)
				print(p2)
				result = model.most_similar(positive=[p,p2],topn=10)
				print("=========RESULT========")
				for result in result:
					print(result)
			except:
				print "Test word not in vocab"
		
