from bllipparser import Tree
from collections import defaultdict
import sys
from utils import open_file

if __name__ == '__main__':
  if len(sys.argv) != 3:
    print 'usage: python create_vocab.py train.gz count'
    sys.exit(0)

  threshold = int(sys.argv[2])
  counts = defaultdict(int)
  with open_file(sys.argv[1]) as f:
    for line in f:
        for word in Tree(line).tokens():
            counts[word.lower()] += 1

  for w, c in counts.iteritems():
    if c > threshold:
      print w
