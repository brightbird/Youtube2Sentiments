#!/usr/bin/python
__author__ = 'ytay2'

'''
Script to persist vectors to models for future use
Vector bins are slow?
'''

import gensim, logging
import os 
from gensim.models import Word2Vec

model = Word2Vec.load_word2vec_format('Dataset/GoogleNews-vectors-negative300.bin', binary=True)  # C binary format
model.save('Models/model'+'_'+'GoogleNews'+'_300')