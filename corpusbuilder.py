'''
File IO class
Parser to parse text and csv files
Renamed to fileio.py to not confuse gensim
@author - tayyi
'''

#utility class to parse text and csv files
from nltk import word_tokenize
import nltk
import csv
import re
import os

class Parser():

	def __init__(self):
		print("Corpus builder initialised!")

	#utility method not used
	def grabCorpusDirectory(self):
		fileList = []
		for file in os.listdir("corpus"):
			if file.endswith(".txt") or file.endswith(".csv"):
				fileList.append(file)
		return fileList

	#Build sentences from list of sentences from model directory containing text files
	#Returns a model list of [sentences] used for training	
	def buildSentences(self):
		"Build Sentences for word2vec model training"
		print("Building Sentences from Corpus directory")
		corpus = []	#Models are combined sentences
		for file in os.listdir("corpus"):
			if file.endswith(".txt"):
				with open('corpus/'+file) as f:
					lines = f.readlines()
					print("Opening document of size:" + str(len(lines)))
					for line in lines:
						line = line.decode('utf-8')
						line = line.lower() #converts all to lowercase
						words = word_tokenize(line)
						sentence = []
						for word in words:
							sentence.append(word)
						corpus.append(sentence)
		print("Corpus size:" + str(len(corpus)) +"sentences")
		return corpus


#To test functionality of parser
if __name__ == "__main__":
	print("Running Parser Module...")
	Parser = Parser()
	Parser.buildSentences()