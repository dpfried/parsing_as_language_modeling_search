from bllipparser import RerankingParser, Tree
from utils import open_file, unkify

import math, sys


def generate_nbest(f):
  nbest = []
  count = 0
  for line in f:
    line = line[:-1]
    if line == '':
      continue
    if count == 0: # the very first
      count = int(line.split()[0])
    elif line.startswith('('):
      nbest.append({'ptb': line})
      count -= 1
      if count == 0:
        yield nbest
        nbest = []


def ptb(line, words, spmrl=False):
  if spmrl:
    line = "(TOP " + line.split(' ', 1)[-1]
  t = Tree(line)
  forms = []
  ptb_recurse(t.subtrees()[0], words, forms, spmrl=spmrl)
  return ' ' + ' '.join(forms) + ' '


def ptb_recurse(t, words, forms, spmrl=False):
  forms.append('(' + t.label)
  for child in t.subtrees():
    if child.is_preterminal():
      token = child.tokens()[0]
      output_token = token if spmrl else token.lower()
      if output_token not in words:
        forms.append('<unk>' if spmrl else unkify(token))
      else:
        forms.append(output_token)
    else:
      ptb_recurse(child, words, forms, spmrl=spmrl)
  forms.append(')' + t.label)


def read_vocab(path):
  words = {}
  for line in open_file(path):
    words[line[:-1]] = len(words)
  return words


def remove_duplicates(nbest):
  new_nbest = []
  for t in nbest:
    good = True
    for new_t in new_nbest:
      if t['seq'] == new_t['seq']:
        good = False
        break
    if good:
      new_nbest.append(t)
  return new_nbest


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("vocab_file", help="vocab.gz")
  parser.add_argument("gold_file", help="gold.gz")
  parser.add_argument("nbest_file", nargs="?", help="nbest.gz")
  parser.add_argument("--spmrl", action="store_true")

  args = parser.parse_args()
  # if len(sys.argv) != 3 and len(sys.argv) != 4:
  #   print 'usage: python traversal.py vocab.gz gold.gz [nbest.gz]'
  #   sys.exit(0)

  #words = read_vocab(sys.argv[1])
  words = read_vocab(args.vocab_file)
  if not args.nbest_file:
    for line in open_file(args.gold_file):
      print ptb(line[:-1], words, args.spmrl)
  else:
    rrp = RerankingParser()
    parser = 'wsj/WSJ-PTB3/parser'
    rrp.load_parser_model(parser)
    for gold, nbest in zip(open_file(args.gold_file),
                           generate_nbest(open_file(args.nbest_file))):
      for tree in nbest:
        tree['seq'] = ptb(tree['ptb'], words, args.spmrl)
      nbest = remove_duplicates(nbest)
      gold = Tree(gold)
      print len(nbest)
      for t in nbest:
        scores = Tree(t['ptb']).evaluate(gold)
        print scores['gold'], scores['test'], scores['matched']
        print t['seq']
