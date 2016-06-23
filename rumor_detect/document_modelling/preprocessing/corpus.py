# -*- encoding: utf-8 -*-
# Author: Qiaoan Chen <kazenoyumechen@gmail.com>

import codecs


class Corpus(object):
    """
    Document corpus constructed from a csv file. Each filed in this file should
    be comma seperated and the file should contains a header line.
    Each line of the csv file represents a document and should follow the format
    <bizuin,appmsgid,itemidx,malicioustype,doc>.
    """
    def __init__(self, path):
        """
        Construct corpus from the file <path> and validate the csv file format.
        Input:
        - path: csv input file path.
        """
        self.path_ = path
        with codecs.open(path, "r", "utf-8") as in_file:
            header = in_file.readline().strip()
            assert header=="bizuin,appmsgid,itemidx,malicioustype,doc"

    def __iter__(self):
        """
        Iterate each line in the csv.
        Output:
        - line: a single line in the csv.
        """
        with codecs.open(self.path_, "r", "utf-8") as in_file:
            header = in_file.readline()
            for line in in_file:
                yield line.strip()


class TextCorpus(Corpus):
    """
    Construct corpus from a csv file. This corpus only return _doc_ field in
    the csv file.
    """
    def __init__(self, path):
        Corpus.__init__(self, path)

    def __iter__(self):
        """
        Iterate each line and extract the doc column from the csv.
        Output:
        - doc: a string represents the doc column in each line.
        """
        for line in Corpus.__iter__(self):
            doc = ",".join(line.split(",")[4:])
            yield doc


class TextSegmentedCorpus(TextCorpus):
    """
    Construct corpus from a csv file. This corpus only return _doc_ field in
    the csv file. A return instance is a list of strings and each string is
    a word in the original document.
    """
    def __init__(self, path):
        TextCorpus.__init__(self, path)

    def __iter__(self):
        """
        Iterate each line and extract the doc column from the csv.
        Output:
        - doc: a string represents the doc column in each line.
        """
        for doc in TextCorpus.__iter__(self):
             yield [word for word in doc.split(" ") if len(word.strip()) > 0]
