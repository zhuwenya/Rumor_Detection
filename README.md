# Rumor_Detection

## Text Data Set
We construct three different size of data set(Text-S, Text-M, Text-L) from a
_SecretDataSet_ to verify our experiments.

__Small(Text-S)__: The small size data set contains 18644 rumor articles
and 19999 normal articles.

## Experimental Protocols

__TFIDF n-gram__: This bag-of-words model is constructed by selecting 500,000
most frequent words (up to 2-gram) from the training dataset.
We use the word count as term-frequency. The inverse document frequency is the
logarithm of the division between total number of documents and number of
documents where the word appears. To deal with 0-value idf, idf value is
smoothing by adding 1. After getting document n-gram vector, a logistic
regression is applied to perform classification. We use TFIDFVectorizer and
LogisticRegression provided by [scikit-learn][sklearn].

__doc2vec__: This model is implemented in [gensim][gensim], and is equivalent
to paragraph vector (Le et al., 2014). To better understand doc2vec,
we test PV-DBOW, PV-DM (two variants of paragraph vector) and PV (PV-DBOW and
PV-DM vectors concatenated) performance. The dimension of embedding is fixed to
400 and other hyper parameters are selected by cross validation. After getting
embedding vectors, a logistic regression is applied to perform classification.

__CNN__:CNN is a single hidden layer word based convolution neural
network for classification (Kim, 2014). The original purpose for this
network is to do sentiment analysis. Because of its simpleness and efficiency,
it is a strong baseline method for sentence classification. Here we treat each
document as a long sentence input to this model. The widths of convolution
filter are [2, 3, 4, 5], 100 filters for each width. We train 200-dimensional
SkipGram (Mikolov et al., 2013) vectors from an unlabeded dataset with 770k
documents. Our code is mainly based on [cnn-text-classification-tf][CNN-tf] and
reimplemented to support loading word vectors. To better understand the effect
of pretrain word vectors, we also conduct an experiment on this model without
pretrain vectors.

## Result
| Method | Variant  | Text-S |
|--------|----------|--------|
| TFIDF n-gram | up to 2-gram | 93.2 |
| doc2vec | PV-DBOW | 88.3 |
| doc2vec | PV-DM | 81.2 |
| doc2vec | PV | 88.7 |
| CNN | word vectors from scratch | 93.3 |
| CNN | pretrain word vectors | 95.9 |


## Reference
* sklearn: <http://scikit-learn.org/>
* gensim: <https://radimrehurek.com/gensim/>
* cnn-text-classification-tf:
<https://github.com/dennybritz/cnn-text-classification-tf>
* Quoc Le and Tomas Mikolov. 2014.
Distributed Representations of Sentences and Documents.
* Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado and Jeffrey Dean. 2013.
Distributed Representations of Words and Phrases and their Compositionality.
* Yoon Kim. 2014.
Convolutional Neural Networks for Sentence Classification.



[sklearn]: http://scikit-learn.org/
[gensim]: https://radimrehurek.com/gensim/
[CNN-tf]: https://github.com/dennybritz/cnn-text-classification-tf
