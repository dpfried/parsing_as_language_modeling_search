from __future__ import print_function
from rnng_interpolate import parse_likelihood_file
from rnng_output_to_nbest import parse_rnng_file
import subprocess
import numpy as np
import json
import sys

def parse_rnng_gen_score_file(line_iter, num_parses_by_sent):
    all_scores = []
    for n_parses in num_parses_by_sent:
        scores = []
        last_sent_len = None
        for i in range(n_parses):
            toks = next(line_iter).strip().split()
            sent_len, neg_log_prob = int(toks[0]), float(toks[1])
            assert(last_sent_len is None or sent_len == last_sent_len)
            last_sent_len = sent_len
            scores.append(-neg_log_prob)
        all_scores.append(scores)
    return all_scores

def eval_b(ref_file, out_file):
    command = ["EVALB/evalb", "-p", "EVALB/COLLINS_S1.prm", ref_file, out_file]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE)
    recall = None
    precision = None
    f1 = None
    complete_match = None
    for line in proc.stdout:
        if line.startswith('Bracketing Recall'):
            assert(recall is None)
            recall = float(line.strip().split('=')[1].strip())
        if line.startswith('Bracketing Precision'):
            assert(precision is None)
            precision = float(line.strip().split('=')[1].strip())
        if line.startswith('Bracketing FMeasure'):
            assert(f1 is None)
            f1 = float(line.strip().split('=')[1].strip())
        if line.startswith('Complete match'):
            assert(complete_match is None)
            complete_match = float(line.strip().split('=')[1].strip())
            # return here before reading <= 40
            return (recall, precision, f1, complete_match)

def rescore(all_indices_proposals_and_parses, all_scores1, all_scores2, lambda1, lambda2, ref_file, out_file):
    with open(out_file, 'w') as f:
        for ipp, scores1, scores2 in zip(all_indices_proposals_and_parses, all_scores1, all_scores2):
            (ix, proposal_score, parse), score1, score2 = max(zip(ipp, scores1, scores2),
                        key=lambda ((ix, proposal_score, parse), score1, score2): (1 - lambda1 - lambda2) * proposal_score + lambda1 * score1 + lambda2 * score2)
            f.write(parse)
            f.write("\n")
    return eval_b(ref_file, out_file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rnng_samples_file", required=True)
    parser.add_argument("--lstm_score_file", required=True)
    parser.add_argument("--rnng_gen_score_file", required=True)
    parser.add_argument("--gold_corpus_file", required=True)
    parser.add_argument("--lstm_lambda", type=float)
    parser.add_argument("--rnng_gen_lambda", type=float)
    parser.add_argument("--lstm_granularity", type=float, default=0.05)
    parser.add_argument("--rnng_gen_granularity", type=float, default=0.05)
    parser.add_argument("--parse_output_file", default="/tmp/_interpolate_parses")
    parser.add_argument("--scores_output")
    args = parser.parse_args()


    with open(args.lstm_score_file) as f:
        lstm_scores = list(parse_likelihood_file(f))

    with open(args.rnng_samples_file) as f:
        rnng_indices_discrim_scores_and_parses = list(parse_rnng_file(f))

    sample_lens = [len(l) for l in rnng_indices_discrim_scores_and_parses]

    with open(args.rnng_gen_score_file) as f:
        rnng_gen_scores = parse_rnng_gen_score_file(f, sample_lens)

    assert(len(lstm_scores) == len(sample_lens))
    assert(len(rnng_gen_scores) == len(sample_lens))
    for n, l_samps, r_samps in zip(sample_lens, lstm_scores, rnng_gen_scores):
        assert(n == len(l_samps))
        assert(n == len(r_samps))

    if args.lstm_lambda is None:
        lstm_lambdas = np.arange(0, 1+1e-3, args.lstm_granularity)
    else:
        lstm_lambdas = [args.lstm_lambda]

    if args.rnng_gen_lambda is None:
        rnng_gen_lambdas = np.arange(0, 1+1e-3, args.rnng_gen_granularity)
    else:
        rnng_gen_lambdas = [args.rnng_gen_lambda]

    scores = []
    N = len(lstm_lambdas) * len(rnng_gen_lambdas)
    i = 0
    for lstm_lambda in lstm_lambdas:
        for rnng_gen_lambda in rnng_gen_lambdas:
            i += 1
            if lstm_lambda + rnng_gen_lambda > 1:
                continue
            sys.stderr.write("\r%d / %d" % (i, N))
            (precision, recall, f1, complete_match) = rescore(rnng_indices_discrim_scores_and_parses, lstm_scores, rnng_gen_scores, lstm_lambda, rnng_gen_lambda, args.gold_corpus_file, args.parse_output_file)
            scores.append({
                'lstm_lambda': lstm_lambda,
                'rnng_gen_lambda': rnng_gen_lambda,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'complete_match': complete_match
            })
    sys.stderr.write("\n")
    if args.scores_output:
        with open(args.scores_output, 'w') as f:
            json.dump(scores, f)
    print(max(scores, key=lambda d: d['f1']))
