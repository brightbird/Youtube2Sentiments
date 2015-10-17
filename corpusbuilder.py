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
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import *

class Parser():

	def __init__(self):
		print("Corpus builder initialised!")

	#utility method not used
	def grabCorpusDirectory(self):
		fileList = []
		for file in os.listdir("Corpus"):
			if file.endswith(".txt") or file.endswith(".csv"):
				fileList.append(file)
		return fileList

	#Build sentences from list of sentences from model directory containing text files
	#Returns a model list of [sentences] used for training	
	def buildSentences(self):
		"Build Sentences for word2vec model training"
		print("Building Sentences from Corpus directory")

		#NLTK Lemmatization and Stemming
		lmtzr = nltk.stem.wordnet.WordNetLemmatizer()
		stemmer = PorterStemmer()

		corpus = []	#Models are combined sentences

		for file in os.listdir("Corpus"):
			if file.endswith(".txt"):
				with open('Corpus/'+file) as f:
					lines = f.readlines()
					print("Opening document of size:" + str(len(lines)))
					for line in lines:
						line = line.decode('utf-8')
						line = line.lower() #converts all to lowercase
						words = word_tokenize(line)
						sentence = []
						for word in words:
							word = stemmer.stem(word)
							word = lmtzr.lemmatize(word)
							sentence.append(word)
						corpus.append(sentence)
		#Handles annotated documents
		for file in os.listdir("Dataset"):
			if file.endswith(".csv"):
				with open('Dataset/'+file) as dataFile:
					patternForSymbol = re.compile(r'(\ufeff)', re.U)
					print("Including annotated documents...")
					reader = csv.reader(dataFile, delimiter=',')
					for index,row in enumerate(reader):
						for index,row in enumerate(reader):
							if(index==0):
								continue
							row[0] = row[0].decode('utf-8')
							rowEdited = re.sub(patternForSymbol, '', row[0])
							comment = rowEdited if rowEdited != "" else row[0]
							sentiment = row[1]
							comment = comment.lower()
							words = word_tokenize(comment)   
							sentence = []
							for word in words:
								word = stemmer.stem(word)
								word = lmtzr.lemmatize(word)
								sentence.append(word)
							corpus.append(sentence)
		print("Corpus size:" + str(len(corpus)) +"sentences")
		return corpus


#To test functionality of parser
if __name__ == "__main__":
	print("Running Parser Module...")
	Parser = Parser()
	Parser.buildSentences()