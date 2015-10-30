__author__="ytay2,espen"

'''
Twitter Crawler
Using Live Streaming API
Please call this script from root directory
'''

import codecs
import sys
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import time
import json
import keys
import os.path


#consumer key, consumer secret, access token, access secret.
#Get your own keys, keys have been omitted
ckey=keys.ckey
csecret=keys.csecret
atoken=keys.atoken
asecret=keys.asecret

class listener(StreamListener):

    def on_data(self, data):
        all_data = json.loads(data)

        if(all_data.get("text", 1) == 1): return True
        tweet = all_data["text"]
        
        username = all_data["user"]["screen_name"]
        #tweetString = tweet.decode("utf-8")
        #usernameString = tweet.decode("utf-8")
        try:
            print((username,tweet))
            output = open("Raw/twitter-out.txt", "a")
            output.write(tweet)
            output.write('\n')
            output.close()
        except(UnicodeEncodeError):
            print(UnicodeEncodeError)
        
        return True

    def on_error(self, status):
        print (status)

#sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
#sys.stdout = codecs.getwriter('utf8')(sys.stdout)
auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["music",'chopin','beethoven','sonata','haydn','bach','mozart','classical','pianist','kyle landry'], languages=['en'])