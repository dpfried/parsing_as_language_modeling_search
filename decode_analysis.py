from __future__ import print_function
import sys
import numpy as np
import ptb_reader
from evaluate import eval_b

from collections import namedtuple

DecodeInstance = namedtuple('DecodeInstance', 'index, gold_linearized, pred_linearized, gold_ptb, pred_ptb, gold_score, pred_score, pred_rescore, match, time')

if __name__ == "__main__":
    output_file = sys.argv[1]
    pred_file = sys.argv[2]

    gold_file = sys.argv[3]

    with open(pred_file) as f:
        pred_ptb_trees = [line.strip() for line in f]

    with open(gold_file) as f:
        gold_ptb_trees = [line.strip() for line in f]

    gold_linearized_trees = []
    pred_linearized_trees = []
    gold_scores = []
    pred_scores = []
    pred_rescores = []
    matches = []
    times = []

    pred_when_pred_better = '/tmp/pred_when_pred_better.out'
    gold_when_pred_better = '/tmp/gold_when_pred_better.out'

    pred_when_gold_better = '/tmp/pred_when_gold_better.out'
    gold_when_gold_better = '/tmp/gold_when_gold_better.out'

    pred_when_tied = '/tmp/pred_when_tied.out'
    gold_when_tied = '/tmp/gold_when_tied.out'

    pred_all = '/tmp/pred.out'
    gold_all = '/tmp/gold.out'

    with open(output_file) as f:
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
                gold_linearized_trees.append(line.split("\t")[1].strip())
            elif line.startswith("pred:"):
                pred_linearized_trees.append(line.split("\t")[1].strip())
            elif line.strip().endswith("seconds") and not line.startswith("DECODE"):
                times.append(float(line.split()[0]))

    num_sents = len(gold_scores)
    assert(len(gold_scores) == num_sents)
    assert(len(pred_scores) == num_sents)
    assert(len(pred_rescores) == num_sents)
    assert(len(matches) == num_sents)
    assert(len(times) == num_sents)
    assert(len(gold_ptb_trees) == num_sents)
    assert(len(pred_ptb_trees) == num_sents)

    decode_instances = [DecodeInstance(i, *args)
                        for i, args in enumerate(zip(gold_linearized_trees,
                                                     pred_linearized_trees,
                                                     gold_ptb_trees,
                                                     pred_ptb_trees,
                                                     gold_scores,
                                                     pred_scores,
                                                     pred_rescores,
                                                     matches,
                                                     times))]
    assert(len(decode_instances) == num_sents)

    def count(predicate):
        return len([di for di in decode_instances if predicate(di)])

    score_and_rescore_absdiff = np.mean([abs(di.pred_score - di.pred_rescore) for di in decode_instances])

    num_matches = count(lambda di: di.match)

    def pred_is_higher(di):
        return di.pred_score > di.gold_score + 1e-3

    def gold_is_higher(di):
        return di.pred_score < di.gold_score - 1e-3

    def pred_gold_tied(di):
        return not pred_is_higher(di) and not gold_is_higher(di)

    num_pred_score_is_higher = count(pred_is_higher)
    num_pred_score_matches = count(pred_gold_tied)
    num_gold_score_is_higher = count(gold_is_higher)

    num_mismatches_and_pred_score_is_higher = count(lambda di: pred_is_higher(di) and not di.match)

    assert(num_pred_score_is_higher + num_gold_score_is_higher + num_pred_score_matches == num_sents)

    def frac_str(count):
        return "%d / %d\t(%0.2f%%)" % (count, num_sents, float(count) * 100 / num_sents)

    print("average time:\t%0.2f seconds" % np.mean(times))
    print("n_sents:\t%s" % num_sents)
    print("avg score/rescore diff mag:\t%s" % score_and_rescore_absdiff)
    print("num matches:    \t" + frac_str(num_matches))
    print("num NOT matches:\t" + frac_str(num_sents - num_matches))
    print("num pred > gold:\t" + frac_str(num_pred_score_is_higher))
    print("num pred == gold:\t" + frac_str(num_pred_score_matches))
    print("num pred < gold:\t" + frac_str(num_gold_score_is_higher))

    print("num NOT match and pred > gold:\t" + frac_str(num_mismatches_and_pred_score_is_higher))

    def partitioned_eval(pred_file, gold_file, predicate):
        with open(pred_file, 'w') as f_pred, open(gold_file, 'w') as f_gold:
            for di in decode_instances:
                if not predicate(di):
                    continue
                f_pred.write("%s\n" % di.pred_ptb)
                f_gold.write("%s\n" % di.gold_ptb)

        return eval_b(gold_file, pred_file)

    print()

    print("pred > gold (R, P, F1, exact match):")
    print(partitioned_eval(pred_when_pred_better, gold_when_pred_better, pred_is_higher))
    print("pred == gold (R, P, F1, exact match):")
    print(partitioned_eval(pred_when_tied, gold_when_tied, pred_gold_tied))
    print("pred < gold (R, P, F1, exact match):")
    print(partitioned_eval(pred_when_gold_better, gold_when_gold_better, gold_is_higher))
    print("overall (R, P, F1, exact match):")
    print(partitioned_eval(pred_when_gold_better, gold_when_gold_better, lambda di: True))

