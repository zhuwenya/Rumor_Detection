# Rumor_Detection

## Text Data Set
We construct three different size of data set(Text-S, Text-M, Text-L) from a
_SecretDataSet_ to verify our experiments, each contains about 40k, 400k
and 4000k articles.

__Small(Text-S)__: The small size data set contains 18644 rumor articles
and 19999 normal articles.

__Middle-sized(Text-M)__:

__Large scale(Text-L)__:

## Experimental Protocols

__TFIDF n gram__: This bag-of-words model is constructed by selecting 500,000
most frequent words (up to 2-gram) from the training dataset.
We use the word count as term-frequency. The inverse document frequency is the
logarithm of the division between total number of documents and number of
documents where the word appears. To deal with 0-value idf, idf value is
smoothing by adding 1.

__doc2vec__: This model is implemented in gensim, and is equivalent to
_paragraph vector_. To better understand doc2vec, we test PV-DBOW and PV-DM
 (variants of paragraph vector) performance. The dimension of embedding is
 fixed to 400 and other hyper parameters are selected by cross validation.
 After getting embedding vectors, a logistic regression is applied to perform
 classification.


## Result
| Method  | Text-S | Text-M | Text-L |
|---------|--------|--------|--------|
| TFIDF 2-gram | 93.18 | | | |
| PV-DBOW(doc2vec) | 88.3 | | | |
| PV-DM(doc2vec) | 81.2 | | | |
| PV(doc2vec) | 88.7 | | | |

