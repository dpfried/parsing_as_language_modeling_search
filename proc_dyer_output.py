from __future__ import print_function
import fileinput
import sys

def proc_line(line):
    toks = line.strip().split('|||')
    return int(toks[0].strip()), "(S1 %s)" % toks[2].strip()

if __name__ == "__main__":

    parse_counts = []
    last_ix = None
    sent_parses = []

    def print_parses():
        N = len(sent_parses)
        parse_counts.append(N)
        print(N)
        for parse in sent_parses:
            print(parse)

    for line in fileinput.input():
        ix, parse = proc_line(line)
        if last_ix is not None and ix != last_ix:
            print_parses()
            sent_parses = []
        sent_parses.append(parse)
        last_ix = ix
    print_parses()
    if any(x != parse_counts[0] for x in parse_counts):
        sys.stderr.write("warning: not all sents have same number of parses!\n")
