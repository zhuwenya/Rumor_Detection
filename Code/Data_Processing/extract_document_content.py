# -*- encoding: utf-8 -*-
# Author: Qiaoan Chen <kazenoyumechen@gmail.com>

import codecs
import re
import jieba
from argparse import ArgumentParser
from bs4 import BeautifulSoup


def get_text_parts(line):
    """
    Get text parts from line.

    Input:
    - line: each line is a document in the origin csv file.
    - text: a single line string represents doc title and content.
    """
    text = ",".join(line.split(",")[4:])
    return text


def remove_html(html_content):
    """
    Removing html tags in content.

    Input:
    - html_content: a single line of html.
    - doc_content: a single line of text containing content strings in html_content.
    """
    soup = BeautifulSoup(html_content, "html5lib")
    doc_content = " ".join(soup.strings)
    return doc_content


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
    
    for i, line in enumerate(in_file):
        text = get_text_parts(line)
        doc_content = remove_html(text)
        seg_content = segmentation(doc_content)
        out_file.write(seg_content)
        out_file.write("\n")

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
