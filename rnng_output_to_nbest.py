from __future__ import print_function
import fileinput
import sys

def proc_line(line):
    toks = line.strip().split('|||')
    return int(toks[0].strip()), float(toks[1].strip()), toks[2].strip()

def parse_rnng_file(line_iter):
    last_ix = None
    parse_counts = []
    last_ix = None
    sent_parses = []
    for line in line_iter:
        ix, score, parse = proc_line(line)
        if last_ix is not None and ix != last_ix:
            yield sent_parses
            parse_counts.append(len(sent_parses))
            sent_parses = []
        sent_parses.append((ix, score, parse))
        last_ix = ix
    yield sent_parses
    if any(x != parse_counts[0] for x in parse_counts):
        sys.stderr.write("warning: not all sents have same number of parses!\n")


if __name__ == "__main__":
    for sent_parses in parse_rnng_file(fileinput.input()):
        print(len(sent_parses))
        for ix, score, parse in sent_parses:
            print("(S1 %s)" % parse)
