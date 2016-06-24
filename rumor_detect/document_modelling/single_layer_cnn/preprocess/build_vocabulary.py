# coding=utf-8
# author: Qiaoan Chen <kazenoyumechen@gmail.com>
from argparse import ArgumentParser

from rumor_detect.document_modelling.preprocessing.corpus import TextSegmentedCorpus
from rumor_detect.document_modelling.single_layer_cnn.preprocess.vocabulary import Vocabulary

if __name__ == "__main__":
    """
    Build vocabulary from a list of csv files.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "-csv_files",
        help="a list of csv files",
        nargs="+"
    )
    parser.add_argument(
        "-min_df",
        help="document frequency less than min_df words will be discard."
    )
    parser.add_argument(
        "-vocab_path",
        help="vocabulary save location"
    )
    args = parser.parse_args()

    readers = [TextSegmentedCorpus(csv_file) for csv_file in args.csv_files]
    vocab = Vocabulary(lower_case=True)
    vocab.build(readers, min_df=int(args.min_df))
    vocab.save(args.vocab_path)

    print "Number of words in vocabulary:", vocab.number_words()
