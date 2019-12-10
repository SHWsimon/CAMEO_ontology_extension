
#!/usr/bin/env python
# coding: utf-8

import json
import keras
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def readFile(path):
	if 'train' in path:
		sentence = []
		label = []
		with open(path, 'r') as inputFile:
			for line in inputFile:
				j = json.loads(line)
				sentence.append(j['sentence'])
				label.append(j['label'][0:3])
			dictionary = {"sentence": sentence, "label": label}
			df = pd.DataFrame(dictionary)
			return df
	else:
		sentence = []
		sourceCode = []
		targetCode = []
		with open(path, 'r') as inputFile:

			for lines in inputFile:
				j = json.loads(lines)
				for i in range(0, len(j)):
					sentence.append(j[i]['sentence'])
					sourceCode.append(j[i]['sourceCode'])
					targetCode.append(j[i]['targetCode'])
			dictionary = {"sentence": sentence, "sourceCode": sourceCode, "targetCode": targetCode}
			df = pd.DataFrame(dictionary)
			return df

if __name__ == '__main__':

	# PRETRAIN MODEL
	####################### read file #######################

	codedFile = 'train.txt'
	codedFile = 'codedSentences.txt'
	notCodedFile = 'notCodedSentences.txt'
	print('reading file ...')
	trainDF = readFile(codedFile)
	testDF = readFile('notCodedSentences_copy.txt')
	print(trainDF)
	print(testDF)

	print('seperating data ...')
	train, test = train_test_split(trainDF, test_size=0.1)

	###################### prepocess #######################
	# we using bag of word to prepocess the input data

	print('bag of word ...')
	vectorizer = CountVectorizer(max_features = 300)
	vectorizer.fit(trainDF.get("sentence"))
	xTrain = vectorizer.transform(trainDF.get("sentence"))
	yTrain = trainDF.label.values
	saveVect = open('vectorizer', 'wb')
	pickle.dump(vectorizer, saveVect)     # Saving bag of word for future data use.
	saveVect.close()
	print('transforming ...')
	xTrain = vectorizer.transform(train.get("sentence"))
	yTrain = train.label.values
	xTest = vectorizer.transform(test.get("sentence"))
	yTest = test.label.values

	####################### training #######################
	# we using logistic regression model as our model

	print('training ...')
	logisticRegression = LogisticRegression()
	logisticRegression.fit(xTrain, yTrain)

	####################### save pretrained model #######################
	print('writing file ...')
	saveFile = open('model', 'wb')
	pickle.dump(logisticRegression, saveFile)    # Saving model so we can use it in the future. Don't need to retrain the model. It takes about 8 hours for training this model!
	saveFile.close()

	####################### testing #######################
	print('loading file and predicting ...')
	loadFile = open('model', 'rb')	#load pretrained model
	model = pickle.load(loadFile)
	predict = model.predict(xTest)
	loadFile.close()

	####################### print result #######################
	print('print report ...')
	print(classification_report(yTest, predict))


	####################### extracting patttern #######################
