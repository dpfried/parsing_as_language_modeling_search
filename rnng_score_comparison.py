import rnng_interpolate
import rnng_output_to_nbest
import rnng_threeway_interpolate
import rnng_decode_analysis
import json

import sys

from ptb_reader import get_tags_tokens_lowercase, remove_dev_unk

import evaluate

def output_to_stderr(line):
    sys.stderr.write("%s\n" % str(line))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rnng_stdout_file", required=True)
    parser.add_argument("--rnng_decode_file", required=True)
    parser.add_argument("--rnng_gold_file", required=True)
    parser.add_argument("--rnng_discrim_ptb_samples_file", required=True, help="one candidate per line: ix ||| discrim_score ||| parse with XX and UNK-ed")
    parser.add_argument("--rnng_gen_score_file", required=True, help="one candidate per line: ix ||| discrim_score ||| parse with XX and UNK-ed")
    parser.add_argument("--output_decodes_prefix")
    args = parser.parse_args()

    with open(args.rnng_discrim_ptb_samples_file) as f:
        # [(ix, discrim_score, parse with XX and Unked)]
        rnng_indices_discrim_scores_and_parses = list(rnng_output_to_nbest.parse_rnng_file(f))

    sample_lens = [len(l) for l in rnng_indices_discrim_scores_and_parses]

    N_sents = len(sample_lens)

    with open(args.rnng_gen_score_file) as f:
        rnng_gen_scores = list(rnng_threeway_interpolate.parse_rnng_gen_score_file(f, sample_lens))

    assert(len(rnng_gen_scores) == N_sents)
    for (p, s) in zip(rnng_indices_discrim_scores_and_parses, rnng_gen_scores):
        assert(len(p) == len(s))

    # pred_ptb of each decode_instance contains un-XXed and un-Unked pares
    decode_instances = rnng_decode_analysis.parse_decode_output_files(args.rnng_gold_file, args.rnng_decode_file, args.rnng_stdout_file)

    with open(args.rnng_gold_file) as f:
        gold_trees = [l.strip() for l in f]

    for i, (tree, di) in enumerate(zip(gold_trees, decode_instances)):
        assert(di.gold_ptb == tree)

    assert(len(decode_instances) == N_sents)

    best_proposal_fname = '/tmp/best_from_proposal.out'
    best_proposal_gold_fname = '/tmp/best_from_proposal_and_gold.out'
    best_proposal_decode_fname = '/tmp/best_from_proposal_and_decode.out'
    best_proposal_gold_decode_fname = '/tmp/best_from_proposal_gold_and_decode.out'

    only_decode_fname = '/tmp/only_decode.out'
    only_gold_fname = '/tmp/only_gold.out'

    def percent_tuple_str(n):
        return "%s / %s (%0.2f%%)" % (n, N_sents, float(n) * 100 / N_sents)

    parses = []

    decode_beats_proposal = 0
    decode_beats_gold = 0
    decode_beats_gold_and_proposal = 0
    # just the samples

    decode_ge_proposal = 0

    best_from_proposal = [
        max((rnng_gen_score, parse) for (_, _, parse), rnng_gen_score in zip(ipp, scores))
        for ipp, scores, gold_tree in zip(rnng_indices_discrim_scores_and_parses, rnng_gen_scores, gold_trees)
    ]

    best_from_proposal = [
        (score, remove_dev_unk(gold_tree, parse)) for (gold_tree, (score, parse)) in zip(gold_trees, best_from_proposal)
    ]

    if args.output_decodes_prefix:
        decode_output_file = open(args.output_decodes_prefix, 'w')
    else:
        decode_output_file = None

    with open(best_proposal_fname, 'w') as f_proposal,    open(best_proposal_gold_fname, 'w') as f_proposal_gold,    open(best_proposal_decode_fname, 'w') as f_proposal_decode,    open(best_proposal_gold_decode_fname, 'w') as f_proposal_gold_decode,    open(only_decode_fname, 'w') as f_decode, open(only_gold_fname, 'w') as f_gold:
        for i, (best_proposal_score, best_proposal) in enumerate(best_from_proposal):
            # best_proposal = best_proposal.replace("*HASH*", '#')
            gold_parse = gold_trees[i]
            decode_instance = decode_instances[i]

            # note: the tags in the decode_instance.gold_ptb are the actual gold
            # tags, not the ones predicted by the POS tagger, b/c of the data
            # fed to the LSTM
            gold_tags, gold_tokens, _ = get_tags_tokens_lowercase(gold_parse)
            assert(gold_tokens == get_tags_tokens_lowercase(decode_instance.gold_ptb)[1])

            decode_pred_tags, decode_pred_tokens, _ = get_tags_tokens_lowercase(decode_instance.pred_ptb)

            assert(decode_pred_tokens == gold_tokens)
            assert(decode_pred_tags == gold_tags)
            # assert(len(decode_pred_tags) == len(lstm_pred_tokens))
            # assert(len(gold_tags) == len(gold_tokens))

            if decode_output_file is not None:
                output_string = remove_dev_unk(gold_parse, decode_instance.pred_ptb)
                # for ix, (gold_tag, gold_token) in enumerate(zip(gold_tags, gold_tokens)):
                #     to_rep = '(XX %s)' % gold_token
                #     assert(to_rep in output_string)
                #     output_string = output_string.replace(to_rep, '(%s %s)' % (gold_tag, gold_token), 1)
                decode_output_file.write("%s\n" % output_string.replace('#', '*HASH*'))

            f_proposal.write("%s\n" % best_proposal)

            f_gold.write("%s\n" % decode_instance.gold_ptb)
            f_decode.write("%s\n" % decode_instance.pred_ptb)

            if decode_instance.pred_score > best_proposal_score + 1e-3:
                decode_beats_proposal += 1

            if decode_instance.pred_score >= best_proposal_score - 1e-3:
                decode_ge_proposal += 1

            if decode_instance.pred_score > decode_instance.gold_score + 1e-3:
                decode_beats_gold += 1

            if decode_instance.pred_score > decode_instance.gold_score + 1e-3 and decode_instance.pred_score > best_proposal_score + 1e-3:
                decode_beats_gold_and_proposal += 1

            f_proposal_decode.write("%s\n" % max([
                        (best_proposal_score, best_proposal),
                        (decode_instance.pred_score, decode_instance.pred_ptb)
                    ], key=lambda t: t[0])[1])

            f_proposal_gold.write("%s\n" % max([
                        (best_proposal_score, best_proposal),
                        (decode_instance.gold_score, decode_instance.gold_ptb)
                    ], key=lambda t: t[0])[1])

            f_proposal_gold_decode.write("%s\n" % max([
                        (best_proposal_score, best_proposal),
                        (decode_instance.pred_score, decode_instance.pred_ptb),
                        (decode_instance.gold_score, decode_instance.gold_ptb)
                    ], key=lambda t: t[0])[1])

    if args.output_decodes_prefix is not None:
        decode_output_file.close()
        with open(args.output_decodes_prefix + ".json", 'w') as f:
            json.dump([di._asdict() for di in decode_instances], f)

    output_to_stderr("decode beats proposal:\t" + percent_tuple_str(decode_beats_proposal))
    output_to_stderr("decode beats gold:\t" + percent_tuple_str(decode_beats_gold))
    output_to_stderr("decode beats gold and prop:\t" + percent_tuple_str(decode_beats_gold_and_proposal))
    output_to_stderr("decode >= proposal:\t" + percent_tuple_str(decode_ge_proposal))
    output_to_stderr("")
    output_to_stderr("gold sanity check (R, P, F1, exact match):")
    output_to_stderr(evaluate.eval_b(args.rnng_gold_file, only_gold_fname))
    output_to_stderr("only decode (R, P, F1, exact match):")
    output_to_stderr(evaluate.eval_b(args.rnng_gold_file, only_decode_fname))
    output_to_stderr("rescore proposal (R, P, F1, exact match):")
    output_to_stderr(evaluate.eval_b(args.rnng_gold_file, best_proposal_fname))
    output_to_stderr("rescore proposal+gold (R, P, F1, exact match):")
    output_to_stderr(evaluate.eval_b(args.rnng_gold_file, best_proposal_gold_fname))
    output_to_stderr("rescore proposal+decode (R, P, F1, exact match):")
    output_to_stderr(evaluate.eval_b(args.rnng_gold_file, best_proposal_decode_fname))
    output_to_stderr("rescore proposal+decode+gold (R, P, F1, exact match):")
    output_to_stderr(evaluate.eval_b(args.rnng_gold_file, best_proposal_gold_decode_fname))
