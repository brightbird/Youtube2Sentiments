#!/usr/bin/python
__author__ = 'ytay2'

'''
Word2Vec Module for NLP Project
Command Line Script to Train a Word2Vec model
Uses gensim (python word2vec implementation)
Supports selecting data source for training word2vec model
Builds everything from Corpus dir ./Corpus
'''

# import libraries, modules
import gensim, logging
import os 
from nltk import word_tokenize
import nltk
import csv
import re

#own modules 
from corpusbuilder import Parser

#logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#main script execution
if __name__ == "__main__":

	print("Running word2vec Module...+\n")
	print("Choose Corpus to train from:")
	choice = raw_input("Select your data source:\n 1) Own data (Corpus Directory) \n 2) Google News Dataset \n 4) Quit \n")
	print(choice)
	if(choice=='1'):
		#build corpus for model training
		print("Training data from own corpus")
		identifier = "music"
		Parser = Parser()
		corpus = Parser.buildSentences()
		#Train model
		model = gensim.models.Word2Vec(corpus, min_count=2,workers=4,size=300)
		print("Completed training model")
		#Saving model
		model.save('Models/model'+'_'+identifier+'_L')
		print("Persisted model to /Models directory")
	elif(choice=='2'):
		print("Building corpus from Google News")
		model = gensim.models.Word2Vec.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
		print("Completed training model")
		model.save('Models/model'+'_'+'Google'+'_L')
		print("Persisted model to /Models directory")
	else:
		print("Error input! Quitting")

