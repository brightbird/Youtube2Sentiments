#!/usr/bin/python
__author__ = 'ytay2'


'''
Builder for File IO
Parser to parse text and csv files from directories
Renamed to corpusBuilder.py to not confuse gensim (it has it's own parser import)
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
from nltk.corpus import stopwords
# encoding=utf8
import sys


'''
Class - Parser
This handles several things
1. Building corpus from raw directory
2. Building sentences (input format) for Word2Vec model
3. Reveal corpus statistics (from raw->cleaned)
'''
class Parser():

	def __init__(self):
		print("Corpus builder initialised!")
		reload(sys)
		sys.setdefaultencoding('utf8')

	#utility method not used
	def grabCorpusDirectory(self):
		fileList = []
		for file in os.listdir("Corpus"):
			if file.endswith(".txt") or file.endswith(".csv"):
				fileList.append(file)
		return fileList

	#Aggregates all raw text files into corpus 
	#From Raw Dir to Corpus directory (aggregated file)
	def buildCorpus(self):
		print("Cleaning and building Corpus file....")
		corpus = open('Corpus/corpus.txt', 'w')

		#regex checkers
		removeLinks = re.compile(r'https?[^\s]*')
		removeRetweets = re.compile(r'rt')
		patternForSymbol = re.compile(r'(\ufeff)', re.U)

		#statistic variables
		linksRemoved = 0
		retweetsRemoved = 0 
		totalTokenCount = 0 
		corpusSize = 0
		rejectedDocs = 0 
		numericalSpam = 0
		duplicates = 0
		lastWritten = ''

		#actual action
		for file in os.listdir("Raw"):
			if file.endswith(".txt"):
				with open('Raw/'+file) as f:
					lines = f.readlines()
					print("Opening document of size:" + str(len(lines)))
					for line in lines:
						line = line.decode('utf-8')
						line = line.lower() #converts all to lowercase
						line = re.sub(patternForSymbol, '', line) #emoji seperate
						if(line==lastWritten):
							print("Removing duplicated spam")
							duplicates+=1
							continue
						if re.findall(removeRetweets, line):
							print("Removing retweet!")
							retweetsRemoved+=1
							continue
						if re.findall(removeLinks, line):
							print("Removing link!")
							linksRemoved+=1
							continue
						numbers = sum(c.isdigit() for c in line)
						words   = sum(c.isalpha() for c in line)
						#spaces  = sum(c.isspace() for c in line)
						if(numbers>words or (numbers>(0.2*len(line)))):
							print('Rejecting numerical spam')
							numericalSpam+=1
							continue
						words = word_tokenize(line)
						totalTokenCount+=len(words)
						if (len(words)<10 or len(line)<100):
							print("Rejecting short content")
							rejectedDocs+=1
							continue
						corpus.write(line)
						lastWritten = line
						corpusSize+=1
		print("Finished processing")
		print("--------------------Preprocessing Statistics-------------------")
		print("Total lines written:"+str(corpusSize))
		print("Total Token Count:" + str(totalTokenCount))
		print('Average token count:'+str(totalTokenCount/corpusSize))
		print("Rejected Short Length Docs:" + str(rejectedDocs))
		print("Removed Retweets:" + str(retweetsRemoved))
		print("Removed Links:" + str(linksRemoved))
		print("Removed Numerical Spam:" + str(numericalSpam))
		print("Removed Duplicates:" + str(duplicates))



	#Build sentences from list of sentences from model directory containing text files
	#Returns a model list of [sentences] used for training	
	def buildSentences(self,lemmatize,stem):
		"Build Sentences for word2vec model training"
		print("Building Sentences from Corpus directory....")

		corpus = []	#Models are combined sentences

		#Statistic Variables
		totalWordCount = 0
		retweetsRemoved = 0 
		linksRemoved = 0 

		#NLTK Lemmatization and Stemming
		if (lemmatize or stem):
			lmtzr = nltk.stem.wordnet.WordNetLemmatizer()
			stemmer = PorterStemmer()
		
		for file in os.listdir("Corpus"):
			if file.endswith(".txt"):
				with open('Corpus/'+file) as f:
					lines = f.readlines()
					print("Opening document of size:" + str(len(lines)))
					for line in lines:
						line = line.decode('utf-8')
						words = word_tokenize(line)
						sentence = []
						#Eliminate stop words
						words = [w for w in words if not w in stopwords.words("english")]
						wordCount = len(words)
						for word in words:
							if(stem):word = stemmer.stem(word)
							if(lemmatize):word = lmtzr.lemmatize(word)
							sentence.append(word)
						corpus.append(sentence)
						totalWordCount+=wordCount

		#Handles annotated documents as well (Youtube comments that are annotated)
		#Keeping preprocssing as it is first
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
							words = [w for w in words if not w in stopwords.words("english")]
							sentence = []
							for word in words:
								if(stem):word = stemmer.stem(word)
								if(lemmatize):word = lmtzr.lemmatize(word)
								sentence.append(word)
							corpus.append(sentence)
		averageSentenceLength = totalWordCount / len(corpus) 
		print("------------------------------Corpus Statistics---------------------------")
		print("Corpus size:" + str(len(corpus)) +"sentences")
		print("Average Sentence Length:" + str(averageSentenceLength))
		return corpus


#To test functionality of parser as a component
if __name__ == "__main__":
	print("Running Parser Module...")
	Parser = Parser()
	Parser.buildCorpus()
	#Parser.buildSentences(True,True)