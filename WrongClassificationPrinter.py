def printWrongClassifiedComments(testLabels, predictionLabels, testData):
    print("MissClassifications ")
    print("Comment ---------------------- Prediction | True value")
    for i in range(len(testLabels)):
        if testLabels[i] != predictionLabels[i]:
            print("Misclassified: ", testData[i].encode('ascii', 'ignore'), predictionLabels[i], testLabels[i])
        else:
            print("Correctly classified: ", testData[i].encode('ascii', 'ignore'), predictionLabels[i], testLabels[i])
