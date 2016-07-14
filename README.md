# Rumor_Detection

## Data Preprocess
__Small(Text-S)__: ...

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

__CNN__: CNN is a single hidden layer word based convolution neural
network for classification (Kim, 2014). The original purpose for this
network is to do sentiment analysis. Because of its simpleness and efficiency,
it is a strong baseline method for sentence classification. Here we treat each
document as a long sentence input to this model. The widths of convolution
filter are 2, 3, 4 and 5, 64 filters for each width. We train 100-dimensional
SkipGram (Mikolov et al., 2013) vectors from an unlabeded dataset with 770k
documents. Our code is mainly based on [cnn-text-classification-tf][CNN-tf] and
reimplement it to support loading word vectors. To better understand the effect
of pretrain word vectors, we also conduct an experiment on this model without
pretrain vectors.

__LSTM__: This model begins with a look-up table that creates a embedding
representation of each words and transforms the input sentence into a three
dimensional tensor of shape _b_ x _t_ x _h_, with _b_ the instances batch size,
_t_ the max length of input sequences and _h_ the dimension of word embedded
space. Different methods are experimented to get a good hidden vectors of
input sequences. A softmax layer is applied to the hidden vectors to perform
classification. In our experiements, we found that bi-directional LSTM
implementation in tensorflow is inefficient to our problem such that we can't
successfully train a model.

## Text-S Result
| Method | Variant  | AUC Score | Precision-0.9 | Batch Time(128 instance) /s | Memory /M |
|--------|----------|-----------|---------------|-----------------------------|-----------|
| TFIDF n-gram | up to 2-gram | 93.11 | 82.1 | - | - |
| doc2vec | PV-DBOW | - | - | - | - |
| doc2vec | PV-DM | - | - | - | - |
| doc2vec | PV | - | - | - | - |
| CNN | word vectors from scratch | - | - | - | - |
| CNN | pretrain word vectors | - | - | - | - |
| ResNet | 18 layer | - | - | - | - |
| LSTM | last hidden vector | - | - | - | - |
| LSTM | mean pooling hidden vector | - | - | - | - |
| Bi-LSTM | mean pooling hidden vector | - | - | - | - |

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
