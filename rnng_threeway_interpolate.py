from __future__ import print_function
from rnng_interpolate import parse_likelihood_file
from rnng_output_to_nbest import parse_rnng_file
import numpy as np
import json
import sys
from evaluate import eval_b
from ptb_reader import get_tags_tokens_lowercase

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

def rescore(all_indices_proposals_and_parses, all_scores1, all_scores2, lambda1, lambda2, ref_file, out_file):
    with open(out_file, 'w') as f:
        for ipp, scores1, scores2 in zip(all_indices_proposals_and_parses, all_scores1, all_scores2):
            (ix, proposal_score, parse), score1, score2 = max(zip(ipp, scores1, scores2),
                        key=lambda ((ix, proposal_score, parse), score1, score2): (1 - lambda1 - lambda2) * proposal_score + lambda1 * score1 + lambda2 * score2)
            f.write(parse)
            f.write("\n")
    return eval_b(ref_file, out_file)

def flatten(lol):
    return [x for l in lol for x in l]

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
    parser.add_argument("--lstm_decode_rnng_reordered_file", help="ptb-format (with HASH replaced) decode trees, in rnng order, produced by rnng_score_comparison")
    parser.add_argument("--lstm_decode_instances_rnng_reordered_file", help="json-serialized decode instances, in rnng order, json format, produced by rnng_score_comparison")
    parser.add_argument("--lstm_decode_rnng_discrim_scores_file", help="rnng discrim scores for the decode instances")
    parser.add_argument("--lstm_decode_rnng_gen_scores_file", help="rnng gen scores for the decode instances")
    args = parser.parse_args()

    with open(args.lstm_score_file) as f:
        lstm_scores = list(parse_likelihood_file(f))

    with open(args.rnng_samples_file) as f:
        rnng_indices_discrim_scores_and_parses = list(parse_rnng_file(f))

    sample_lens = [len(l) for l in rnng_indices_discrim_scores_and_parses]

    N_sents = len(sample_lens)
    print("%d sentences" % N_sents)

    with open(args.rnng_gen_score_file) as f:
        rnng_gen_scores = parse_rnng_gen_score_file(f, sample_lens)

    assert(len(lstm_scores) == N_sents)
    assert(len(rnng_gen_scores) == N_sents)

    for n, l_scores, r_scores in zip(sample_lens, lstm_scores, rnng_gen_scores):
        assert(n == len(l_scores))
        assert(n == len(r_scores))

    if args.lstm_decode_rnng_reordered_file:
        with open(args.lstm_decode_rnng_reordered_file) as f:
            decode_parses = [l.strip() for l in f]
        assert(len(decode_parses) == N_sents)

        with open(args.lstm_decode_instances_rnng_reordered_file) as f:
            decode_instances = json.load(f)
        assert(len(decode_instances) == N_sents)

        with open(args.lstm_decode_rnng_discrim_scores_file) as f:
            decode_rnng_indices_discrim_scores_and_parses = list(parse_rnng_file(f))
        assert(all(len(l) == 1 for l in decode_rnng_indices_discrim_scores_and_parses))
        decode_rnng_discrim_scores = [t[1] for t in flatten(decode_rnng_indices_discrim_scores_and_parses)]
        assert(len(decode_rnng_discrim_scores) == N_sents)

        with open(args.lstm_decode_rnng_gen_scores_file) as f:
            decode_rnng_gen_scores = flatten(parse_rnng_gen_score_file(f, [1] * N_sents))
        assert(len(decode_rnng_gen_scores) == N_sents)

        for i, (decode_parse,\
                decode_instance,\
                decode_rnng_discrim_score,\
                decode_rnng_gen_score,\
                candidate_indices_discrim_scores_and_parses,\
                candidate_lstm_scores,\
                candidate_gen_scores) in enumerate(zip(decode_parses,
                                                       decode_instances,
                                                       decode_rnng_discrim_scores,
                                                       decode_rnng_gen_scores,
                                                       rnng_indices_discrim_scores_and_parses,
                                                       lstm_scores,
                                                       rnng_gen_scores)):
            decode_tags, decode_tokens, _ = get_tags_tokens_lowercase(decode_parse)
            cand_tags, cand_tokens, _ = get_tags_tokens_lowercase(candidate_indices_discrim_scores_and_parses[0][2])
            assert(decode_tokens == cand_tokens)
            assert(decode_tags == cand_tags)
            # tags may differ b/c of rnng processing
            assert(decode_tokens == get_tags_tokens_lowercase(decode_instance['pred_ptb'].replace('#', '*HASH*'))[1])
            assert(i == candidate_indices_discrim_scores_and_parses[0][0])
            candidate_indices_discrim_scores_and_parses.append((i, decode_rnng_discrim_score, decode_parse))
            candidate_lstm_scores.append(decode_instance['pred_score'])
            candidate_gen_scores.append(decode_rnng_gen_score)
            # test to make sure gets the same as decode scores
            # candidate_indices_discrim_scores_and_parses[:] = [(i, decode_rnng_discrim_score, decode_parse)]
            # candidate_lstm_scores[:] = [decode_instance['pred_score']]
            # candidate_gen_scores[:] = [decode_rnng_gen_score]


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
