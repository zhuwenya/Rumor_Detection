# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 15:30:23 2016

@author: wenya
"""

from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(path):
    corpus=[]
    for line in open(path): 
        corpus.append(line.lower())
    vectorizer = CountVectorizer(min_df=1)
    X = vectorizer.fit_transform(corpus)
    Y=X.toarray()
    dictionary=vectorizer.get_feature_names()
    index2word={}
    for word in dictionary:
        index=vectorizer.vocabulary_.get(word)
        index2word[index]=word
    return Y,dict

if __name__=="__main__": 
    path='/home/wenya/code/Wechat_Rumor_530/B2W/mycorpus.txt'
    feature,index2word=bag_of_words(path)
    
