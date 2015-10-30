from nltk import word_tokenize


def splitIntoThreeEqualTokenSet(dataSet): #Dataset format: [{comment:Comment ...
    comments = []
    tokenCounter = 0
    tokensComulativeCount = []
    for i, instance in enumerate(dataSet):
        comments.append(instance["comment"])
        tokens = word_tokenize(instance["comment"])
        for word in tokens:
            if len(word) < 2: continue
            tokenCounter += 1
        tokensComulativeCount.append(tokenCounter)
    tokensPerSet = tokensComulativeCount[-1] / 3
    set1End = findIndex(tokensComulativeCount, tokensPerSet)
    set2End = findIndex(tokensComulativeCount, tokensPerSet*2)
    set1 = dataSet[0:set1End]
    set2 = dataSet[set1End:set2End]
    set3 = dataSet[set2End:]
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