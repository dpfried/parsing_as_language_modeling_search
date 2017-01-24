import sys
import numpy as np

if __name__ == "__main__":
    decode_file = sys.argv[1]

    golds = []
    preds = []
    gold_scores = []
    pred_scores = []
    pred_rescores = []
    matches = []
    times = []

    with open(decode_file) as f:
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
                golds.append(line.split("\t")[1].strip())
            elif line.startswith("pred:"):
                preds.append(line.split("\t")[1].strip())
            elif line.strip().endswith("seconds") and not line.startswith("DECODE"):
                times.append(float(line.split()[0]))

    num_sents = len(gold_scores)
    assert(len(gold_scores) == num_sents)
    assert(len(pred_scores) == num_sents)
    assert(len(pred_rescores) == num_sents)
    assert(len(matches) == num_sents)
    assert(len(times) == num_sents)

    score_and_rescore_absdiff = []

    num_matches = 0

    num_pred_score_is_higher = 0
    num_pred_score_matches = 0
    num_pred_score_is_lower = 0

    num_mismatches_and_pred_score_is_higher = 0

    for gold, pred, gold_score, pred_score, pred_rescore, match, time in zip(golds, preds, gold_scores, pred_scores, pred_rescores, matches, times):
        score_and_rescore_absdiff.append(abs(pred_score - pred_rescore))
        if match:
            num_matches += 1

        pred_is_higher = pred_score > gold_score + 1e-3
        pred_is_lower = pred_score < gold_score - 1e-3

        if pred_is_higher:
            num_pred_score_is_higher += 1
            assert(not match)
        elif pred_is_lower:
            num_pred_score_is_lower += 1
            assert(not match)
        else:
            num_pred_score_matches += 1
            assert(match)

        if not match and pred_is_higher:
            num_mismatches_and_pred_score_is_higher += 1

    assert(num_pred_score_is_higher + num_pred_score_is_lower + num_pred_score_matches == num_sents)

    def frac_str(count):
        return "%d / %d\t(%0.2f%%)" % (count, num_sents, float(count) * 100 / num_sents)

    print("average time:\t%0.2f seconds" % np.mean(times))
    print("n_sents:\t%s" % num_sents)
    print("num matches:    \t" + frac_str(num_matches))
    print("num NOT matches:\t" + frac_str(num_sents - num_matches))
    print("num pred > gold:\t" + frac_str(num_pred_score_is_higher))
    print("num pred = gold:\t" + frac_str(num_pred_score_matches))
    print("num pred < gold:\t" + frac_str(num_pred_score_is_lower))

    print("num NOT match and pred > gold:\t" + frac_str(num_mismatches_and_pred_score_is_higher))
