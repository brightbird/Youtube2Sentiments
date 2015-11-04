import gensim
import re
import featureExtractorW2V
from pickler import getPickleObject

__author__ = 'espen1'


class ContextExtractor():
    def __init__(self, model, tryExtraSimWords = True, acceptContextThreshold = 0.25):
        self.model = model
        self.videoSimilarityWords = ["video"]
        self.songSimilarityWords = ["song"]
        if tryExtraSimWords:
            featureExtractorW2V.addSimilarWordsToVocab(len(self.videoSimilarityWords), model, self.videoSimilarityWords, wordCount=10)
            featureExtractorW2V.addSimilarWordsToVocab(len(self.songSimilarityWords), model, self.songSimilarityWords, wordCount=10)
        self.acceptContextThreshold = acceptContextThreshold


    #Confidence is defined as: ratio between video & song similarity, r. Confidence = (1-r) * the size of the similarity
    def calculateConfidences(self, topVideo, topSong):
        #Index0 = video, Index1 = song
        confidences = []
        labels = []
        for videoScore, songScore in zip(topVideo, topSong):
            videoSongList = [videoScore, songScore]
            if featureExtractorW2V.ownMax(videoSongList) == 0: ##Both values are 0
                confidences.append(0)
            elif videoScore == 0 or songScore == 0: #One of the values is 0
                confidences.append(max(videoSongList))
            else: ##Both values greater than 0
                ratio = videoSongList[0] / videoSongList[1]  if videoSongList[0] < videoSongList[1] else videoSongList[1] / videoSongList[0] #Highest term in denominator
                confidences.append(max(videoSongList) * (1 - ratio)) #If both values are the same ratio would be 1 and confidence = 0
            label = "video" if videoScore > songScore else "song"
            if confidences[-1] < self.acceptContextThreshold:
                label = "other"
            labels.append(label)
        return labels, confidences

    def preProcessComments(self, comments):
        asciiComments = []
        for comment in comments:
            asciiComments.append(comment.encode('ascii', 'ignore'))
        processedComments = []
        for comment in asciiComments:
            processedComments.append(self.preProcessComment(comment))
        return processedComments

    #Splits all words that ends with a symbol instead of a whitespace
    def preProcessComment(self, comment):
        editedComment = ""
        lastSpan = 0
        anyWordLetterFollowedBySympolPattern = re.compile(r'\w[^\w\s]')
        matchObject = re.search(anyWordLetterFollowedBySympolPattern, comment[lastSpan:])
        while matchObject:
            span = matchObject.span()
            editedComment += comment[lastSpan: lastSpan+ span[0] + 1] + " "
            lastSpan += span[1] -1
            matchObject = re.search(anyWordLetterFollowedBySympolPattern, comment[lastSpan:])
        if lastSpan != 0:
            editedComment += comment[lastSpan:]
        if editedComment == "":
            return comment
        else:
            return editedComment

    def giveContext(self, comment):
        videoScores = featureExtractorW2V.findSimilarity(comment, self.model, self.videoSimilarityWords, featureExtractorW2V.ownMax)
        songScores = featureExtractorW2V.findSimilarity(comment, self.model, self.videoSimilarityWords, featureExtractorW2V.ownMax)
        labels, confidences = self.calculateConfidences([max(videoScores), max(songScores)])
        return labels[0], confidences[0] #E.g. "Video", 0.523511

    def getContexts(self, comments):
        comments = self.preProcessComments(comments)
        maxScoresVideo = []
        maxScoresSong = []
        for comment in comments:
            maxScoresVideo.append(featureExtractorW2V.findSimilarity(comment, self.model, self.videoSimilarityWords, featureExtractorW2V.ownMax))
            maxScoresSong.append(featureExtractorW2V.findSimilarity(comment, self.model, self.songSimilarityWords, featureExtractorW2V.ownMax))
        topVideoScore = [max(values) for values in maxScoresVideo]
        topSongScore = [max(values) for values in maxScoresSong]
        labels, confidences = self.calculateConfidences(topVideoScore, topSongScore)
        return labels, confidences



if __name__ == "__main__":
    comments = getPickleObject("input/comments")
    model = gensim.models.Word2Vec.load('Models/model28')
    contextExtractor = ContextExtractor(model)
    labelsExtra, confidencesExtra = contextExtractor.getContexts(comments)

    contextExtractorWithoutExtraWords = ContextExtractor(model, False)
    labels, confidences = contextExtractorWithoutExtraWords.getContexts(comments)

    differentLabels = 0
    for i in range(len(comments)):
        if labelsExtra[i] != labels[i]:
            differentLabels += 1
            print(comments[i].encode('ascii', 'ignore'))
            print("Label extra sim words:", labelsExtra[i], " confidence: ", confidencesExtra[i])
            print("Label no extra sim words:", labels[i], " confidence: ", confidences[i])
    ratio = differentLabels / len(comments)
    print("Different labels (%): ", differentLabels/ len(comments), differentLabels, ratio)