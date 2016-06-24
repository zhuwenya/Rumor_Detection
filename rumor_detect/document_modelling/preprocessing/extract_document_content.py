# coding=utf-8
# Author: Qiaoan Chen <kazenoyumechen@gmail.com>

import codecs
import re
import jieba
from argparse import ArgumentParser
from bs4 import BeautifulSoup

class Document:
    """
    A document object representing a line in the csv file.
    For processing convenience, the html parts of the original document are
    discard. Title and content are combine into an attribute named doc.
    """
    def __init__(self, line):
        args = line.split(",")
        self.bizuin_ = args[0]
        self.appmsgid_ = args[1]
        self.itemidx_ = args[2]
        self.malicioustype_ = args[-1]
        doc = ','.join(args[3:-1])
        doc = Document.remove_html(doc)
        doc = Document.segmentation(doc)
        self.doc_ = doc

    def to_csv_line(self):
        return '%s,%s,%s,%s,%s' % (self.bizuin_, self.appmsgid_, self.itemidx_,
                                   self.malicioustype_, self.doc_)

    @staticmethod
    def remove_html(html_content):
        """
        Removing html tags in content.

        Input:
        - html_content: a single line of html.
        - doc_content: a single line of text containing content strings in
          html_content.
        """
        soup = BeautifulSoup(html_content, "html5lib")
        doc_content = " ".join(soup.strings)
        return doc_content

    @staticmethod
    def segmentation(doc_content):
        """
        Segmenting doc_content with jieba.

        Input:
        - doc_content: a single line of text in chinese.
        - seg_content: a single line of text after segmentation.
        """
        seg_list = jieba.cut(doc_content)
        seg_content = " ".join(seg_list)

        # compress multiple spaces into one space
        seg_content = re.sub(u"(\s|　)+", " ", seg_content, flags=re.UNICODE)
        return seg_content


def extract_document_content(in_file, out_file):
    """
    Extract contents in in_file to out_file.

    Input:
    - in_file: input file object.
    - out_file: output file object.
    """
     # skip csv header line
    in_file.readline()
    out_file.write('bizuin,appmsgid,itemidx,malicioustype,doc\n')
    
    for i, line in enumerate(in_file):
        try:
            doc = Document(line.strip())
            out_file.write(doc.to_csv_line())
            out_file.write("\n")
        except Exception:
            print "[WARN] line %d parse error, skip" % (i,)

        if i % 1000 == 0:
            print "processing line", i
        
       
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "in_doc",
        help="location of the input document file, seperated by command"
    )
    parser.add_argument(
        "out_doc",
        help="location of the output document file, each line is a document"
    )
    args = parser.parse_args()

    with codecs.open(args.in_doc, "r", "utf-8") as in_file, \
         codecs.open(args.out_doc, "w", "utf-8") as out_file:
        extract_document_content(in_file, out_file)
