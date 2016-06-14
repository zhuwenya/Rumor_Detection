# -*- encoding: utf-8 -*-

import codecs
import re
from time import sleep
from argparse import ArgumentParser
from bs4 import BeautifulSoup


def extract_document_content(in_file, out_file):
    in_file.readline() # skip csv header line
    for line in in_file:
        raw_content = ",".join(line.split(",")[4:])
        soup = BeautifulSoup(raw_content, "html5lib")
        content = " ".join(soup.strings)
        content = re.sub(u"(\s| | |　)+", " ", content, flags=re.UNICODE)
        out_file.write(content)
        out_file.write("\n")
        
       
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
