from nltk import word_tokenize


def splitIntoThreeEqualTokenSet(dataSet, selectNumberOfComments = 0): #Dataset format: [{comment:Comment ...
    comments = []
    tokenCounter = 0
    tokensComulativeCount = []
    positiveCount = 0
    negativeCount = 0
    if selectNumberOfComments == 0: selectNumberOfComments = len(dataSet)
    for i, instance in enumerate(dataSet):
        if i > selectNumberOfComments-1: break #If we want to only select x amount of comments we do this
        sentiment = instance['sentiment']
        if (sentiment == 'positive'):
            positiveCount += 1
        elif (sentiment == 'negative'):
            negativeCount += 1
        comments.append(instance["comment"])
        tokens = word_tokenize(instance["comment"])
        for word in tokens:
            if len(word) < 2: continue
            tokenCounter += 1
        tokensComulativeCount.append(tokenCounter)
    tokensPerSet = tokensComulativeCount[-1] / 3
    set1End = findIndex(tokensComulativeCount, tokensPerSet)
    set2End = findIndex(tokensComulativeCount, tokensPerSet*2)
    set3End = findIndex(tokensComulativeCount, tokensPerSet*3)
    print("Positive:", positiveCount)
    print("Negative:", negativeCount)
    set1 = dataSet[0:set1End]
    set2 = dataSet[set1End:set2End]
    set3 = dataSet[set2End:len(dataSet) if selectNumberOfComments == len(dataSet) else set3End]
    return [set1, set2, set3]

def findIndex(comulatives, value):
    for i in range(len(comulatives)):
        if value < comulatives[i]:
            return i


def tokenChecker(dataSet):
    tokenCounter = 0
    for instance in dataSet:
        tokens = word_tokenize(instance["comment"])
        for word in tokens:
            if len(word) < 2: continue
            tokenCounter += 1
    print(tokenCounter)