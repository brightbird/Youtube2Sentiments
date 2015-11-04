import pickle
from nltk import word_tokenize
from pickler import getPickleObject

vocabulary = [u"happy", u"positive", u"negative", u"cool", u"nice", u"love", u"hat", u"not", u"poor", u"good", u"ugly", u"handsome", u"fail", u'gay',u'sucked', u'glad',u'bastard',u'fat', u'inspire', u'quality']
wordsSet = False

#Called before getFeatures method to specify vocabulary
def setSimilarityWords(model, words=None, useStoredWords = False):
    global wordsSet, vocabulary
    if wordsSet: return
    extraWords = 60
    if words != None:
        vocabulary = words
    elif useStoredWords:
        negativeWords = getPickleObject("input/negatives")
        positiveWords = getPickleObject("input/positives")
        vocabulary = list(set(addSimilarWordsToVocab(len(negativeWords), model, negativeWords, False) + addSimilarWordsToVocab(len(positiveWords), model, positiveWords)))
    else:
        vocabulary = list(set(addSimilarWordsToVocab(extraWords, model, vocabulary)))
    wordsSet = True


def addSimilarWordsToVocab(extraWords, model, vocabulary, positive=True, wordCount = 5):
    for word in vocabulary:
        if extraWords < 1: break
        try:
            similars = model.most_similar(positive=[word], topn=wordCount) if positive else model.most_similar(negative=[word], topn=wordCount)
            for word, sim in similars:
                vocabulary.append(word)
        except KeyError:
            pass
        extraWords -= 1
    return vocabulary

def getFeatures(comments, model, similarityWords = None):
    if similarityWords == None:
        similarityWords = vocabulary
    features = []
    for i, comment in enumerate(comments):
        featureValue = findSimilarity(comment, model, similarityWords)
        features.append(featureValue)
    return features

def average(listOfValues):
    if len(listOfValues) == 0: return 0
    return sum(listOfValues) / len(listOfValues)
def ownMax(listOfValues):
    if len(listOfValues) == 0: return 0
    return max(listOfValues)
def findSimilarity(comment, model, similarityWords, combineFunction = None):
    words = word_tokenize(comment)
    if combineFunction == None:
        combineFunction = average #Will by default average the similarities
    vocabularyScores = []
    for similarityWord in similarityWords:
        simScores = []
        for word in words:
            try:
                if len(word) < 2: continue
                simScores.append(model.similarity(similarityWord, word))
            except KeyError:
                pass
        vocabularyScores.append(combineFunction(simScores))
    return vocabularyScores
