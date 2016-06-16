import pandas as pd
import sklearn
import numpy 
from numpy import *
from sklearn import svm
import re

Punctuations=[u"\u3001",u"\u3002",u"\uFF0C",u"\uFF1F",u"\uFF01",u"\u300E",u"\u300F",u"\uFF08",u"\uFF09",u"\u3014",u"\u3015",u"\uFF1A",u"\uFF1B",u"\u300C",u"\u300D",u"\u2018",u"\u2019",u"\u201C",u"\u201D",u"\uFF0E"]
Rumor_content=pd.read_csv("F:\Rumor_data\Rumor_content_sample_1000.csv",encoding="gbk",delimiter="#",error_bad_lines=False)
y=Rumor_content.iloc[900].tolist()
# transform the list to the string
str1 = ''.join(y)
pat=re.sub('<p>',"。".decode('utf-8'),str1)
# filteringt1-filtering the english and mainly store the chinese character
# but still some special character will be stored
p = re.compile("[A-Za-z0-9\[\`\~\!\@\#\$\^\&\*\(\)\=\|\{\}\'\:\;\'\,\[\]\.\<\>\/\?\~\！\@\#\\\&\*\%\"\-\_\s]", re.L)
Document_content=pd.DataFrame( columns=['document_id', 'document_content', 'document_flag','document_len'])
x=p.sub("",pat)
if len(x) != 0:
    L=""
    for e in x:
        if u"\u4e00"<=e<=u"\u9fa5":
            L=L+e
        if e in Punctuations:
            L=L+e
    if L[-1] not in [u"\u3001","\u3002",u"\uFF0C",u"\uFF1F",u"\uFF01"]:
        L=L+u"\u3002"
    if L[0] == u"\u3002":
        L=L[1:-1]
    #统计字数
     len(L)


"""
Rumor_contex_category=Rumor_contex['malicioustype']

number_category=pd.value_counts(Rumor_contex_category)

number_category.plot(kind='bar')
