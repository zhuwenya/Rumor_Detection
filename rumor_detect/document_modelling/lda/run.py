# coding=utf-8
# author: Qiaoan Chen <kazenoyumechen@gmail.com>


import codecs
import logging

from gensim.corpora import Dictionary
from gensim.models import LdaModel

logger = logging.getLogger('lda.run')


class UnlabeledCorpus(object):
    def __init__(self, path, vocab):
        self.path_ = path
        self.vocab_ = vocab

    def __iter__(self):
        vocab = self.vocab_
        tokens = set(vocab.values())

        with codecs.open(self.path_, 'r', 'utf-8') as in_f:
            for line in in_f:
                doc = [word for word in line.strip().split()
                       if len(word) > 0 and word in tokens]
                doc = vocab.doc2bow(doc)
                if len(doc) > 0:
                    yield doc


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )
    vocab = Dictionary.load_from_text('./vocab.txt')
    corpus = UnlabeledCorpus('./rumor_train.csv', vocab)
    valid_corpus = UnlabeledCorpus('./rumor_valid.csv', vocab)
    valid_sentences = [doc for doc in valid_corpus][5000:]

    # varing number of topics
    # result = {}
    # for num_topics in [2, 4, 8, 16, 32, 64]:
    #     best_value = -100
    #     for i in range(5):
    #         model = LdaModel(corpus=corpus, id2word=vocab, num_topics=num_topics)
    #         likelihood = model.log_perplexity(valid_sentences)
    #         best_value = max(best_value, likelihood)
    #     result[num_topics]= best_value
    #
    # for num_topics, likelihood in result.iteritems():
    #     print 'num_topics: %d, best word_likelihood: %f' % (num_topics, likelihood)

    model = LdaModel(corpus=corpus, id2word=vocab, num_topics=8, passes=2)
    model.save('./lda_model.txt')
    # print topics to a file
    topics = model.show_topics(num_topics=100, num_words=50)
    with codecs.open('./topics.txt', 'w', 'utf-8') as out_f:
        for topic in topics:
            topic_id, topic_str = topic[0], topic[1]
            out_f.write('%d:\n%s\n' % (topic_id, topic_str))
        out_f.write('\n')



