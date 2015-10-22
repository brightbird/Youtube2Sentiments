#!/usr/bin/python
__author__ = 'espen1,ytay2'

'''
Naive Classifier Module for NLP Project
Written by Espen Albert, Tay Yi

Usage - run Python naive.py in CLI (does not support argparser yet)

Uses Naive Classifier for Sentiment Analysis
Poor Accuracy - 60-70% :( 

'''

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
 

#Global Controls
#Not used yet 
VERBOSE = False 
DEBUG = False 


#Build adjective feature set
def addAdjectivesToFeatureSet(tagged):
    for tag in tagged:
        if tag[1] in adjectives:
            adjectiveWords.append(tagged[0][0].lower())


#Refactored to single function that passes in feature choice instead of 2 seperate functions
def findFeatures(comment,featureChoice):
    words = word_tokenize(comment)
    features = {}
    for w in featureChoice:
        features[w] = w in words
    return features

#Subset construction for training data 
#Ommitted. Is there really a need to wrap it in objects? 
#Cause many enumeration issues later on 
class Subset:
    def __init__(self, featureSets, start, end):
        self.featureSet = featureSets[start:end]
        self.start = start
        self.end = end
    def getFeatureSet(self):
        return self.featureSet
    def setPrecisionValue(self, value):
        self.precision = value
    def setRecallValue(self, value):
        self.recall = value
    def setFValue(self, value):
        self.recall = value
    def setErrorSentences(self, value):
        self.errorSentences = value

    def __iter__(self):
        return iter(self.featureSet)      #Allows subset list to be iterable 

#I think there are nltk packages to calculate recall and precision - ty 
def testClassifier(classifier, testSet):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    #i = 0 + testSet
    errorSentences = []
    for i,comment in enumerate(testSet):
        classifyValue = classifier.classify(comment[0])
        if classifyValue == 'positive':
            if comment[1] == classifyValue:
                tp+= 1
            else:
                fp += 1
                errorSentences.append("False positive error on sentence: " +  comments[i][0])
        else:
            if comment[1] == classifyValue:
                tn += 1
            else:
                fn += 1
                errorSentences.append("False negative error on sentence: "+ comments[i][0])

    '''
    testSet.setErrorSentences(errorSentences)
    precision = tp / (tp + fp)
    testSet.setPrecisionValue(precision)
    print("Precision: "+ str(precision))
    recall = tp / (tp + fn)
    testSet.setRecallValue(recall)
    print("Recall: "+ str(recall))
    fMeasure = 2 * tp / (tp + fn)
    testSet.setFValue(fMeasure)
    print("F-measure: "+ str(fMeasure))
    '''
    
    print("Number of documents tested:" + str(len(testSet)))
    print("Number of errors:" + str(len(errorSentences)))
    '''
    for error in errorSentences:
        print(error)
    '''

#Build Subsets based on number of partitions
#Inputs -> Featureset a list [] of features from different subsets
#Output -> Wrapping everything into a subset object and returns it 
def SubsetConstruction(featureset,partition):
    dataSets = []
    incrementor = len(featureset)/ partition
    print(incrementor)
    start_window = 0
    end_window = incrementor
    subset_size = 0 
    for i in range(0,partition):
        #built partitions
        tempSubset = featuresets[start_window:end_window]
        dataSets.append(tempSubset)
        subset_size = len(tempSubset)
        start_window += incrementor
        end_window +=incrementor
    print("Subsets constructed from partitions:" + str(partition) + " of size:" + str(subset_size))
    return dataSets

#Takes in a dataset list[], and trains them and output results with a variety of classfiers 
def trainAndTest(dataSets,partition):

    for i in range(len(dataSets)):
        print("===============PARTITION"+str(i+1)+"================")
        testSet = dataSets[i]
        trainingSet = []
        "Everything that is not the testSet is the training set"
        for index,data in enumerate(dataSets):
            #I'm not sure if this actually returns the featuree-set when iterated upon
            if (index==i):
                print("Skipping..")
                continue
            trainingSet += data

        #print(trainingSet)
        #print("length of training set:" + str(len(trainingSet)))

        "Training and testing steps"
        classifier = nltk.NaiveBayesClassifier.train(trainingSet)
        testClassifier(classifier, testSet)
        print("Naive Bayes Algo accuracy:" , (nltk.classify.accuracy(classifier, testSet))*100)
        classifier.show_most_informative_features(15)
      
        '''
        LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
        LogisticRegression_classifier.train(trainingSet)
        testingSet = testSet
        print("Logisitic regression Algo accuracy:" , (nltk.classify.accuracy(LogisticRegression_classifier, testingSet))*100)

        MNB_classifier = SklearnClassifier(MultinomialNB())
        MNB_classifier.train(trainingSet)
        print("MNB Algo accuracy:" , (nltk.classify.accuracy(MNB_classifier, testingSet))*100)
        '''

def writeTrainingDataToFile(dataSets):
    text_file = open("features.txt", "w")
    for dataset in dataSets:
        for data in dataset:
            print(str(data[0]) + "\n")
            #print(data[0])
            text_file.write(str(data[0]))
    text_file.close()
    print("written features.txt file")
        
#main script execution
if __name__ == "__main__":

    print("Running NLP Module...")
    patternForSymbol = re.compile(r'(\ufeff)', re.U)
    comments =[]

    #reads in CSV file
    with open('Dataset/dataset.csv','rb') as dataFile:
        reader = csv.reader(dataFile, delimiter=',')
        for row in reader:
            row[0] = row[0].decode('utf-8')
            rowEdited = re.sub(patternForSymbol, '', row[0])
            comment = rowEdited if rowEdited != "" else row[0]
            sentiment = row[1]
            comments.append((comment, sentiment))


    comments.pop(0) #Take away the first element that specifies text, sentiment
    adjectives = ["JJ", "JJS", "JJR"]

    adjectiveWords = []
    allWords = []

    #Builds dictionary, feature sets etc...
    for comment in comments:
        words = word_tokenize(comment[0])   
        #words = [word for word in words if word not in stopwords.words('english')]
        for word in words:
            allWords.append(word.lower())
        tags = nltk.pos_tag(words)
        addAdjectivesToFeatureSet(tags)

    print("# Words: ", len(set(allWords)))
    print("# Documents", len(comments))
    wordsFrequencies = nltk.FreqDist(adjectiveWords)
    wordsFrequencies2 = nltk.FreqDist(allWords)
    #print("The most common adjectives" , wordsFrequencies.most_common(100))
    #print("The most common words" , wordsFrequencies2.most_common(100))
    featuresets = []
    wordFeatures =adjectiveWords
    wordFeatures2 =wordsFrequencies2.keys()
    for (comment, sentiment) in comments:
        featuresets.append((findFeatures(comment,wordFeatures2), sentiment))
    #NOTE : partition values should be the same! 
    dataSets = SubsetConstruction(featuresets,3)
    #writeTrainingDataToFile(dataSets);
    trainAndTest(dataSets, 3)
    




