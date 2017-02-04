from __future__ import print_function
import sys
from collections import namedtuple
from ptb_reader import remove_dev_unk

from decode_analysis import analyze

DecodeInstance = namedtuple('DecodeInstance', 'index, gold_ptb, pred_ptb, gold_xx, pred_xx, gold_score, pred_score, pred_rescore, match, time')

def parse_decode_output_files(gold_file, decode_file, stdout_file):
    with open(gold_file) as f:
        gold_ptb_trees = [line.strip() for line in f]

    with open(decode_file) as f:
        pred_xx_trees_stdout = [line.strip() for line in f]

    gold_xx_trees = []
    pred_xx_trees = []
    gold_scores = []
    pred_scores = []
    pred_rescores = []
    matches = []
    times = []

    with open(stdout_file) as f:
        for line in f:
            if line.startswith("gold score"):
                gold_scores.append(float(line.split("\t")[1]))
            elif line.startswith("pred score"):
                pred_scores.append(float(line.split("\t")[1]))
            elif line.startswith("pred rescore"):
                pred_rescores.append(float(line.split("\t")[1]))
            elif line.startswith("match?"):
                matches.append(line.split("\t")[1].strip() == "True")
            elif line.startswith("gold:"):
                gold_xx_trees.append(line.split("\t")[1].strip())
            elif line.startswith("pred:"):
                pred_xx_trees.append(line.split("\t")[1].strip())
            elif line.strip().endswith("seconds") and "DECODE" not in line and '/' not in line:
                times.append(float(line.split()[0]))

    num_sents = len(gold_scores)
    assert(len(gold_scores) == num_sents)
    assert(len(pred_scores) == num_sents)
    assert(len(pred_rescores) == num_sents)
    assert(len(matches) == num_sents)
    assert(len(times) == num_sents)
    assert(len(gold_ptb_trees) == num_sents)
    assert(len(pred_xx_trees) == num_sents)
    assert(len(gold_xx_trees) == num_sents)
    assert(pred_xx_trees_stdout == pred_xx_trees)

    pred_ptb_trees = [remove_dev_unk(gold_ptb, pred_xx)
                      for gold_ptb, pred_xx in zip(gold_ptb_trees, pred_xx_trees)]


    decode_instances = [DecodeInstance(i, *args)
                        for i, args in enumerate(zip(gold_ptb_trees,
                                                     pred_ptb_trees,
                                                     gold_xx_trees,
                                                     pred_xx_trees,
                                                     gold_scores,
                                                     pred_scores,
                                                     pred_rescores,
                                                     matches,
                                                     times))]
    assert(len(decode_instances) == num_sents)
    return decode_instances

if __name__ == "__main__":
    stdout_file = sys.argv[1]
    decode_file = sys.argv[2]
    gold_file = sys.argv[3]

    # decode file shouldn't strictly be necessary, as info in it should be in
    # stdout file, but have it just as a check
    decode_instances = parse_decode_output_files(gold_file, decode_file, stdout_file)
    analyze(decode_instances)
