import urllib2
import re
from pickler import dumpPickle


def getHtml(url):
    headers = {}
    headers['User-Agent'] = "Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17"
    request =urllib2.Request(url, headers=headers)
    resp = urllib2.urlopen(request)
    return resp.read()


def extractNegativeWords():
    url = "http://www.enchantedlearning.com/wordlist/negativewords.shtml"
    return extractWords(url)


def extractWords(url):
    html = getHtml(url)
    words = re.findall(r'\w{2,20}<BR>', html)
    editedWords = []
    for word in words:
        editedWords.append(word[:-4].encode("utf-8"))
    return editedWords[:-1]

def extractPositiveWords():
    url = "http://www.enchantedlearning.com/wordlist/positivewords.shtml"
    return extractWords(url)

if __name__ == "__main__":
    negativeWords = extractNegativeWords()
    positiveWords = extractPositiveWords()
    print("Negatives: ", negativeWords)
    print("Positives: ", positiveWords)
    dumpPickle("../input/negatives", negativeWords)
    dumpPickle("../input/positives", positiveWords)