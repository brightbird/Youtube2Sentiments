import re

import nltk

verbTags = []
getVerbTagsPattern = re.compile(r'VB.?')
def getTags():
    tagFile = open("tagSet.txt","r").readlines()
    getTagPattern = re.compile(r'[A-Z]{2,3}')
    tags = set()
    for line in tagFile:
        if not re.findall(getTagPattern, line):
            continue
        span = re.search(getTagPattern, line).span()
        tag = line[span[0]:span[1]]
        tags.add(tag)
        if re.findall(getVerbTagsPattern, line):
            verbTags.append(tag)
    return tags
tags = getTags()


filename = "Corpus/moreoutput"
inputFile = open("%s.txt" % filename,"r")
allLines = inputFile.readlines()
inputFile.close()
#removeAuthorPattern = re.compile(r'^\(\'[^\']*\', (\'|\")')
#removeEndOfLine = re.compile(r'(\'|\")\)$')
removeRetweets = re.compile(r'RT')
removeLinks = re.compile(r'https?[^\s]*')
editedLines = []
#lines = ["She was in love with a man", "I would like to climb the highest mountain in the world"]



recognizeBaseNounPhrases = re.compile(r'DT(JJ[S])?NN')
chunkGram = r"""Chunk: {<DT><JJ[S]>?<NN[P]?>} """
chunkParser = nltk.RegexpParser(chunkGram)
adjectives = ["JJ"]

def converTagsToString(tagged):
    a = ""
    for tag in tagged:
        a += tag[1]
    return a


def informativeSentence(tagged):
    if not aVerbInSentence(tagged): return False
    chunked = chunkParser.parse(tagged)
    a = chunked.subtrees()
    if re.findall(removeRetweets, line):
         return False
    if not aAdjectiveInSentence(tagged):
        return False
    return True

def aAdjectiveInSentence(tagged):
    for tag in tagged:
        if tag[1] in adjectives:
            return True
    return False

def aVerbInSentence(tagged):
    aVerbInSentence = False
    for tag in tagged:
        if tag[1] in verbTags:
            aVerbInSentence = True
            break
    return aVerbInSentence


def testLine(line):
    line = re.sub(removeLinks, '',line)
    tokens = nltk.tokenize.word_tokenize(line)
    tagged = nltk.pos_tag(tokens)
    if informativeSentence(tagged):
        editedLines.append(line)
    #taggedString = converTagsToString(tagged)
    #if re.findall(recognizeBaseNounPhrases, taggedString):
          #  print(re.findall(recognizeBaseNounPhrases, taggedString))

for line in allLines:
    testLine(line)


print("Number of lines = ", len(editedLines))

setOfLines = set(editedLines)
print("Unique lines = ", len(setOfLines))
output = open("%s.txt" % (filename+ "_edited"), "w")

for line in setOfLines:
    output.write(line)

output.close()
