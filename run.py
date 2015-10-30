__author__ = 'ytay2'

'''
This is the final script
Creates features via word2vec and passes it to SVM for classification
Results are still terribly bad for now.
Uses word2vec average from music model trained from word2vec
'''

# import libraries, modules
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


# Global Controls
LEMMATIZE = True
STEMMING = True
TFIDF = False
WORD2VEC = False
BOW = False
SAVED = True  # True means we do not want to save
saveVectorizer = False
COMBINE = False
W2W_SIM_SENTIMENT = True
"Preprocessing Variables"
patternForSymbol = re.compile(r'(\ufeff)', re.U)  # Regex Emoticon Cleaner
lmtzr = WordNetLemmatizer()
stemmer = PorterStemmer()

# Loop & Testing Controls
iteration = 3

# Word2Vec settings
size = 300  # feature size for word2vec model
key_error_rate = 0
vectorCount = 0
entireTextFailed = 0

# Bag of words count vectorizer
BOWvectorizer = CountVectorizer(analyzer="word", \
                                tokenizer=None, \
                                preprocessor=None, \
                                stop_words=None, \
                                ngram_range=(1, 5), \
                                max_features=10000)

# Making tf-idf vectors
TFIDFvectorizer = TfidfVectorizer(min_df=5,
                                  max_df=0.8,
                                  sublinear_tf=True,
                                  use_idf=True)

"Build word vector averages from Word2vec"


def buildWordVector(model, text, size):
    global vectorCount
    global key_error_rate
    global entireTextFailed
    vectorCount += 1
    errorCount = 0
    count = 0
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        if (STEMMING):
            word = stemmer.stem(word)
        if (LEMMATIZE):
            word = lmtzr.lemmatize(word)
        try:
            # print(model[word].shape)
            vec += model[word].reshape((1, size))
            count += 1
        except KeyError:
            errorCount += 1
            continue
    if count != 0:
        vec /= count
        errorRate = (errorCount / (count + errorCount)) * 100
        # print("Error Percentage:"+str(errorRate))
        key_error_rate += errorRate

    if count == 0:
        key_error_rate += 100
        entireTextFailed += 1
    # print("Entire text failed")
    print(vec.shape)
    return vec[0]


def runLinearSVM():
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
    if (score >= 0.85 and SAVED == False):
        # Save Classifier for future use
        print("Saving classifier of score:" + str(score))
        saveClassifier(classifier_linear)
        SAVED = True  # switch, do not want to save multiple classifiers
        saveVectorizer = True  # switch, yes we want to persist vectorizer
    return score


def runRbfSVM():
    # Perform classification with SVM, kernel=rbf
    classifier_rbf = svm.SVC()
    t0 = time.time()
    classifier_rbf.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_rbf = classifier_rbf.predict(test_vectors)
    t2 = time.time()
    time_rbf_train = t1 - t0
    time_rbf_predict = t2 - t1
    # Print results in a nice table
    print("================Results for SVC(kernel)-RBF========")
    print("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
    print(classification_report(test_labels, prediction_rbf))
    print("Accuracy:", accuracy_score(test_labels, prediction_rbf))
    print("\n")


def runLibLinearSVM():
    # Perform classification with SVM, kernel=linear
    classifier_liblinear = svm.LinearSVC()
    t0 = time.time()
    classifier_liblinear.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_liblinear = classifier_liblinear.predict(test_vectors)
    t2 = time.time()
    time_liblinear_train = t1 - t0
    time_liblinear_predict = t2 - t1
    print("================Results for LibLinear SVC========")
    print("Training time: %fs; Prediction time: %fs" % (time_liblinear_train, time_liblinear_predict))
    print(classification_report(test_labels, prediction_liblinear))
    print("Accuracy:", accuracy_score(test_labels, prediction_liblinear))
    print("\n")


def saveClassifier(classifier):
    filename = 'Classifiers/classifier.pkl'
    _ = joblib.dump(classifier, filename, compress=9)
    print("Classifier persisted!")


def createClassifierInput(dataSets, validationSet = 0):
    global index, feature, vector, sentiment, comment
    training1 = dataSets[(validationSet + 1) % 3]
    training2 = dataSets[(validationSet + 2) % 3]
    partition = len(training1) + len(training2) -1
    ListOfFeatures = training1 + training2 + dataSets[validationSet]
    for index, feature in enumerate(ListOfFeatures):
        if (WORD2VEC): vector = feature['word2vec']
        sentiment = feature['sentiment']
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
    # load model
    if (WORD2VEC):
        print("Loading Model..may take some time..please wait!")
        model = gensim.models.Word2Vec.load('Models/model28')
        #model = Word2Vec.load_word2vec_format('Dataset/GoogleNews-vectors-negative300.bin',
        #                                      binary=True)  # C binary format
    print("Building feature sets...")
    model = gensim.models.Word2Vec.load('Models/model28')

    train_data = []
    train_labels = []
    test_labels = []
    test_data = []
    train_vectors = []
    test_vectors = []


    # stores vectors before spliting into training and testing
    ListOfFeatures = []

    print("Reading dataset..")
    # reads in CSV file
    with open('data/dataset.csv', 'r') as dataFile:
        reader = csv.reader(dataFile, delimiter=',')
        for index, row in enumerate(reader):
            if (index == 0):
                print("Skipping header for data file")
                continue

            # Pre-processing
            row[0] = row[0].decode('utf-8')
            rowEdited = re.sub(patternForSymbol, '', row[0])
            comment = rowEdited if rowEdited != "" else row[0]
            sentiment = row[1]
            comment = comment.lower()
            words = word_tokenize(comment)
            words = [w for w in words if not w in stopwords.words("english")]

            # Creating features
            feature = {}  # feature dictionary for filtering later

            # Building Vectors
            if (WORD2VEC):
                vector = buildWordVector(model, words, size)
                feature['word2vec'] = vector

            feature['comment'] = comment  # save raw text for future processing
            feature['sentiment'] = sentiment
            ListOfFeatures.append(feature)

    totalrows = len(ListOfFeatures)


    partition = (totalrows * 2) / 3
    print("total dataset size:" + str(totalrows))
    print("training size:" + str(partition))

    # Max scores for average results
    TFIDF_MAX = 0
    BOW_MAX = 0
    WORD2VEC_MAX = 0
    COMBINE_MAX = 0
    COMBINE_MAX = 0
    W2W_SIM_SENTIMENT_MAX = 0
    # Main loop
    #random.shuffle(ListOfFeatures)
    dataSets = equalDataSetSplitter.splitIntoThreeEqualTokenSet(ListOfFeatures)
    for set in dataSets:
        equalDataSetSplitter.tokenChecker(set)
    for i in range(0, iteration):
        train_data = []
        train_labels = []
        test_labels = []
        test_data = []
        train_vectors = []
        test_vectors = []

        # Constructing actual input for classifier
        createClassifierInput(dataSets, i%3)

        # Runs word2 vec first because we do not want to corrupt the word2vec vector variable
        if (WORD2VEC):
            # Running once for Word2Vec approach (Averaging)
            print("----------Word2vec Approach------------")
            score = runLinearSVM()
            WORD2VEC_MAX += score

        if (COMBINE):
            print("----------TF-IDF + Word2Vec---------")
            temp_train_vectors = TFIDFvectorizer.fit_transform(train_data)
            temp_test_vectors = TFIDFvectorizer.transform(test_data)
            print(temp_train_vectors.shape)
            print(train_vectors[0].shape)
            combined_train_vector = []
            combined_test_vector = []
            for index, vector in enumerate(train_vectors):
                # print(temp_train_vectors[index,:].shape)
                # print(vector.shape)
                temp = sparse.hstack((vector, temp_train_vectors[index, :]))
                print("tempshape:" + str(temp.shape))
                temp = temp.toarray()[0]
                print(temp.shape)
                combined_train_vector.append(temp)
            for index, vector in enumerate(test_vectors):
                temp = sparse.hstack((vector, temp_test_vectors[index, :]))
                # print(temp)
                temp = temp.toarray()[0]
                print(temp.shape)
                combined_test_vector.append(temp)
            train_vectors = combined_train_vector
            test_vectors = combined_test_vector
            print(train_vectors[0].shape)
            score = runLinearSVM()
            COMBINE_MAX += score

        if (TFIDF):
            print("----------Tf-idf Approach------------")
            train_vectors = TFIDFvectorizer.fit_transform(train_data)
            test_vectors = TFIDFvectorizer.transform(test_data)
            score = runLinearSVM()
            TFIDF_MAX += score

        if (BOW):
            print("----------Bag of words--------------")
            train_vectors = BOWvectorizer.fit_transform(train_data)
            # vocab = BOWvectorizer.get_feature_names()
            # print(vocab)
            test_vectors = BOWvectorizer.transform(test_data)
            score = runLinearSVM()
            BOW_MAX += score
            if (saveVectorizer):
                print("Saving vectorizer..")
                joblib.dump(BOWvectorizer, 'Classifiers/vectorizer.pkl', compress=9)
                saveVectorizer = False
        if (W2W_SIM_SENTIMENT):
            print("----------- Word2Vec similarity--------------")
            train_vectors = featureExtractorW2V.getFeatures(train_data, model)
            test_vectors = featureExtractorW2V.getFeatures(test_data, model)
            featureExtractorW2V.testContext(train_data, model)
            score = runLinearSVM()
            W2W_SIM_SENTIMENT_MAX += score
            print("Score: ", score)
            # print("Persisting SVM...")
            # joblib.dump(classifier_linear, 'svm_linear.pkl')

if (WORD2VEC):
    print("==================Word2Vec Model Evaluation===========")
    print("Average Key Error Rate:" + str(key_error_rate / vectorCount) + "%")
    print("Entire Document Fail Rate:" + str((entireTextFailed * 100) / vectorCount) + "%")

print("================Printing Average Results=============")
if (TFIDF): print("TDIF average score:" + str(TFIDF_MAX / iteration))
if (BOW): print("Bag of words avg score:" + str(BOW_MAX / iteration))
if (WORD2VEC): print("Word2Vec Avg score:" + str(WORD2VEC_MAX / iteration))
if (COMBINE): print("COMBINE Avg score:" + str(COMBINE_MAX / iteration))
if (W2W_SIM_SENTIMENT): print("Word 2 vec similarity Avg score:" + str(W2W_SIM_SENTIMENT_MAX / iteration))


# runRbfSVM()
# runLibLinearSVM()
