#author: wenya zhu
#time: 2016-06-12
#code:using the bag 
from gensim import corpora, models, similarities


# the class to extract the B2W of the documents
class MyCorpus(object):
     def __init__(self,path):
         self.path=path # the path to the files which store the document
     def __iter__(self):
         for line in open(self.path):
             # assume there's one document per line, tokens separated by whitespace
             yield dictionary.doc2bow(line.lower().split())



if __name__=="__main__": 
    path='/home/wenya/code/Wechat_Rumor_530/B2W/mycorpus.txt'
    stoplist = set('for a of the and to in'.split())
    #construct the dictionary 
    dictionary = corpora.Dictionary(line.lower().split() for line in open(path))
    # remove stop words and words that appear only once
    stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]
    once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]
    dictionary.filter_tokens(stop_ids + once_ids)
    #remove gaps in id sequence after words that were removed
    dictionary.compactify()
    corpus_memory_friendly = MyCorpus(path)
    #print the bag-of-words
    print(list(corpus_memory_friendly))
    for vector in corpus_memory_friendly:
        print vector
        
    tfidf = models.TfidfModel(corpus_memory_friendly)
    
