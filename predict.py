
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
import string
import nltk
#nltk.download('averaged_perceptron_tagger')
#nltk.download('universal_tagset')
import collections
import math
from nltk.chunk.regexp import *
from rake_nltk import Rake
from nltk.collocations import *
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import *
import sys

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

def removeStopWords(text_tokens):
    stopwords_set = set(stopwords.words("english")).union(set(string.punctuation))
    stopwords_set = stopwords_set.union(set('’')).union(set('“')).union(set('”'))
    result = [word for word in text_tokens if not word in stopwords_set]
    return result

def removePunctuation(sentence):
    punc_set = set(string.punctuation).union(set('“')).union(set('”')).union(set('’'))
    result = ''.join(ch for ch in sentence if ch not in punc_set)
    return result

# Method 1: extract the most frequent n nouns by POS tagging
def Method1(n=3):
    pattern = []
    M1_input = removeStopWords(input_tokens)
    pos_tags = nltk.pos_tag(M1_input, tagset='universal')
    noun_words = []
    for pair in pos_tags:
        if pair[1] == 'VERB':
            noun_words.append(pair[0])
    keywords = collections.Counter(noun_words).most_common(n)
    for pair in keywords:
        #print(pair[0])
        pattern.append(pair[0])
    return pattern

# Method 2: extract multiple keywords from the most frequent n collocations
def Method2(n=3):
    pattern = []
    M2_input = removeStopWords(input_tokens)
    bgam = nltk.collocations.BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(M2_input)
    collocations = finder.nbest(bgam.likelihood_ratio, n)
    for c in collocations:
        #print(' '.join(c))
        pattern.append(' '.join(c))
    return pattern

# Method 3: extract most frequent n noun phrases
def Method3(n=3):
    pattern = []
    M3_input = sent_tokenize(input_text_sen)
    np_grammar = "NP: {<DET>?<ADJ>*<NOUN>+}"
    chunk_parser = RegexpParser(np_grammar)
    NP = []
    for sent in M3_input:
        tokens = word_tokenize(sent)
        tag_tokens = nltk.pos_tag(tokens, tagset='universal')
        result = chunk_parser.parse(tag_tokens)
        for subtree in result.subtrees():
            if subtree.label() == "NP":
                words = " ".join([x for (x, y) in subtree.leaves()])
                NP.append(words)
    keywords = collections.Counter(NP).most_common(n)
    for data in keywords:
        pattern.append(data[0])
    return pattern


# Method 4: rake (rapid automatic keyword extraction)
def Method4(n=3):
    pattern = []
    M4_input = input_text_sen
    r = Rake()
    r.extract_keywords_from_text(M4_input)
    r.get_ranked_phrases()
    for key in r.get_ranked_phrases_with_scores()[:n]:
        pattern.append(removePunctuation(key[1].strip()))
    return pattern


# Method 5: TD-IDF method
def Method5(n=3):
    M5_input = sent_tokenize(input_text_sen)
    pattern = []
    # TF: fij = frequency of term i in sentence j, Nj = total number of words in sentence j
    fij = []
    Nj = []
    TF_result = []
    for sen in M5_input:
        tokens = word_tokenize(sen)
        word_freq = dict(collections.Counter(tokens))
        fij.append(word_freq)
        Nj.append(len(tokens))
    # Compute TF value
    for i in range(len(Nj)):
        tokens = word_tokenize(M5_input[i])
        temp = {}
        for word in tokens:
            temp[word] = fij[i].get(word) / Nj[i]
        TF_result.append(temp)

    # IDF: N = number of sentences, ni = number of sentences mention word i
    N = len(M5_input)
    IDF_result = {}
    for sen in M5_input:
        tokens = word_tokenize(sen)
        for word in tokens:
            ni = 0
            for s in M5_input:
                if word in s:
                    ni += 1
                    continue
            if ni == 0:
                ni+=1
            IDF_result[word] = math.log(N / ni)

    # Compute TF-IDF value
    TF_IDF = {}
    for sent_TF in TF_result:
        for word, value in sent_TF.items():
            TF_IDF[word] = value * IDF_result[word]
    TF_IDF_sort = sorted(TF_IDF.items(), key=lambda x: x[1], reverse=True)
    for word, value in TF_IDF_sort[:n]:
        pattern.append(word)
    return pattern

if __name__ == '__main__':
    # PREDICT LABEL
    ####################### read file #######################
    notCodeFile = 'notCodedSentences_copy.txt'
    testDF = readFile(notCodeFile)
    ####################### prepocess #######################
    loadVec = open('vectorizer', 'rb')
    vectorizer = pickle.load(loadVec)
    xTest = vectorizer.transform(testDF.get('sentence'))
    ######################### testing #######################
    loadFile = open('model', 'rb')  #load pretrained model
    model = pickle.load(loadFile)
    predict = model.predict(xTest)
    loadFile.close()
    print('\n\n')
    print(predict)
    print('\n\n')
    # ####################### extracting patttern in five method#######################
    print('start extracting ...')
    input_text = testDF['sentence'].astype(str).values.tolist()
    i = 0
    for sen in input_text:
        input_tokens = wordpunct_tokenize(sen)
        input_text_sen = sen

        print(sen)
        res1 = []
        new_pattern = Method1()
        res1.append(predict[i])
        res1.append(new_pattern)
        print(res1)
    
        res2 = []
        new_pattern = Method2()
        res2.append(predict[i])
        res2.append(new_pattern)
        print(res2)

        res3 = []
        new_pattern = Method3()
        res3.append(predict[i])
        res3.append(new_pattern)
        print(res3)   

        res4 = []
        new_pattern = Method4()
        res4.append(predict[i])
        res4.append(new_pattern)
        print(res4)   

        res5 = []
        new_pattern = Method5()
        res5.append(predict[i])
        res5.append(new_pattern)
        print(res5)
        print('\n\n')
        i+=1

	# done


