import fileinput
from bllipparser import RerankingParser, Tree
import sys

if __name__ == '__main__':
  rrp = RerankingParser()
  parser = 'wsj/WSJ-PTB3/parser'
  rrp.load_parser_model(parser)
  for sent_count, line in enumerate(fileinput.input()):
    sys.stderr.write("\r%s" % sent_count)
    tokens = Tree(line).tokens()
    nbest = rrp.parse(tokens)
    print len(nbest)
    for tree in nbest:
      print tree.ptb_parse
