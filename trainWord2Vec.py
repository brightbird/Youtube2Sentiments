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
from gensim.models import Word2Vec

#own modules 
from corpusBuilder import Parser

#logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

size=24

#main script execution
if __name__ == "__main__":

	print("Running word2vec Module...+\n")
	#print("Choose Corpus to train from:")
	#choice = raw_input("Select your data source:\n 1) All data (Corpus Directory) \n 2) Others \n 4) Quit \n")
	#print(choice)
	layers = raw_input("Number of output layers")
	size = int(layers)
	#build corpus for model training
	print("Training data from own corpus")
	#identifier = "music"
	Parser = Parser()
	corpus = Parser.buildSentences(False,False)
	#Train model
	model = gensim.models.Word2Vec(corpus, min_count=4,workers=4,size=size)
	print("Completed training model")
	#Saving model
	#model.save('Models/model'+'_'+identifier+'_L')
	model.save('Models/model'+str(size))
	print("Persisted model to /Models directory")

