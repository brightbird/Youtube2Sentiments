#!usr/bin/python
'''
Python Script for SENNA functionality
Uses PracNLPTools for Sementic Role Labeling
'''

import csv
import re
from practnlptools.tools import Annotator
annotator=Annotator()

print("Running Shallow Semantic Parser")
patternForSymbol = re.compile(r'(\ufeff)', re.U)
comments=[]
#reads in CSV file
with open('data/dataset2.csv','rb') as dataFile:
    reader = csv.reader(dataFile, delimiter=',')
    for row in reader:
        #row[0] = row[0].decode('utf-8')
        rowEdited = re.sub(patternForSymbol, '', row[0])
        comment = rowEdited if rowEdited != "" else row[0]
        sentiment = row[1]
        comments.append((comment, sentiment))


for index,comment in enumerate(comments):
	if(index<100):
		print comment[0]
		print(annotator.getAnnotations(comment[0])['srl'])
		print("==========================")
