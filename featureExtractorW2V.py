from nltk import word_tokenize

standardSimWords = ["happy", "positive", "negative", "cool", "nice", "love", "hat", "not", "poor", "good", "ugly", "handsome", "fail"]

def getFeatures(comments, model, similarityWords = standardSimWords):
    features = [[] for x in range(len(comments))]
    #similarityWords = model.vocab.keys()[1:400]
    for i, comment in enumerate(comments):
        stringComment = comment.encode('ascii', 'ignore')
        #stringComment = comment.decode('utf-8')
        words = word_tokenize(stringComment)
        #words = word_tokenize(comment)
        for j, similarityWord in enumerate(similarityWords):
            cumulativeSimilarity = 0
            for word in words:
                try:
                    if len(word) < 2: continue
                    cumulativeSimilarity += model.similarity(similarityWord, word)
                except KeyError:
                    pass
            featureValue = cumulativeSimilarity / len(words) if len(words) > 0 else 0 #Average else 0
            features[i].append(featureValue)
    return features

videoSimilarityWords = ["video", "artist", "song", "other"]

def testContext(comments, model): #TODO: Test, next time: I think a minimum value before labeling is smart. Also note that the words should be utf8.. not ascii
    simScores = getFeatures(comments, model, videoSimilarityWords)
    labels = []
    for score in simScores:
        max = 0
        index = 0
        for i, value in enumerate(score):
            if value > max:
                index = i
                max = value
        labels.append(videoSimilarityWords[index])
