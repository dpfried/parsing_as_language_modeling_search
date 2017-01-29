import rnng_interpolate
import rnng_output_to_nbest
import rnng_threeway_interpolate
import decode_analysis
import json
import re
import random

import sys

from ptb_reader import get_tags_tokens_lowercase

import pandas
import evaluate
from collections import Counter


def output_to_stderr(line):
    sys.stderr.write("%s\n" % str(line))

if __name__ == "__main__":
    import argparse
    rnng_gold_file = '../rnng/corpora/22.auto.clean'

    parser = argparse.ArgumentParser()
    parser.add_argument("--lstm_search_prefixes", nargs="+", help="list of prefixes for files ending in stderr_block--all and decode_block-all")
    parser.add_argument("--lstm_gold_file", required=True)
    parser.add_argument("--rnng_lstm_rescore_file", required=True, help="file produced by running the lstm reranker on the candidates from rnng, e.g. *.ptb_samples.likelihods. parses should be in the same order as rnng_gold_file may be in a different order than the lstm_search_prefixes and lstm_gold_file files")
    parser.add_argument("--rnng_gold_file", default='../rnng/corpora/22.auto.clean')
    parser.add_argument("--rnng_proposal_file", default='dyer_beam/dev_pos_embeddings_beam=100.ptb_samples')
    parser.add_argument("--rnng_gen_rescore_file", default='dyer_beam/dev_pos_embeddings_beam=100.samples.rnng-gen-likelihoods')
    parser.add_argument("--output_reordered_decodes_prefix")

    args = parser.parse_args()

    if len(args.lstm_search_prefixes) > 1:
        assert(args.output_reordered_decodes_prefix is None)

    # RNNG order

    with open(rnng_gold_file) as f:
        rnng_gold_tokens = [get_tags_tokens_lowercase(line)[1] for line in f]
    with open(rnng_gold_file) as f:
        rnng_gold_trees = [line.strip() for line in f]

    with open(args.rnng_proposal_file) as f:
        rnng_indices_discrim_scores_and_parses = list(rnng_output_to_nbest.parse_rnng_file(f))

    sample_lens = [len(l) for l in rnng_indices_discrim_scores_and_parses]

    with open(args.rnng_lstm_rescore_file) as f:
        rnng_lstm_scores = list(rnng_interpolate.parse_likelihood_file(f))

    with open(args.rnng_gen_rescore_file) as f:
        rnng_gen_scores = list(rnng_threeway_interpolate.parse_rnng_gen_score_file(f, sample_lens))

    # LSTM order

    with open(args.lstm_gold_file) as f:
        lstm_gold_tokens = [get_tags_tokens_lowercase(line)[1] for line in f]
    with open(args.lstm_gold_file) as f:
        lstm_gold_trees = [line.strip() for line in f]

    lstm_decode_indices_by_setting = [decode_analysis.parse_decode_output_files(args.lstm_gold_file, search_prefix + ".decode_block--all", search_prefix + ".stderr_block--all")
                                      for search_prefix in args.lstm_search_prefixes]

    def strip_root_and_tags(parse, root):
        assert(parse.startswith("(" + root + " "))
        parse = parse[len(root) + 3:-1]
        return re.sub("\(\S+ (\S+)\)", "(XX \\1)", parse)

    rnng_gold_trees_stripped = [strip_root_and_tags(parse, "TOP") for parse in rnng_gold_trees]
    lstm_gold_trees_stripped = [strip_root_and_tags(parse, "S1") for parse in lstm_gold_trees]

    rnng_indices_to_lstm_indices = [
        lstm_gold_trees_stripped.index(rnng_gt) for rnng_gt in rnng_gold_trees_stripped
    ]

    best_proposal_fname = '/tmp/best_from_proposal.out'
    best_proposal_gold_fname = '/tmp/best_from_proposal_and_gold.out'
    best_proposal_decode_fname = '/tmp/best_from_proposal_and_decode.out'
    best_proposal_gold_decode_fname = '/tmp/best_from_proposal_gold_and_decode.out'

    only_decode_fname = '/tmp/only_decode.out'
    only_gold_fname = '/tmp/only_gold.out'

    N_sents = len(rnng_indices_discrim_scores_and_parses)

    def percent_tuple_str(n):
        return "%s / %s (%0.2f%%)" % (n, N_sents, float(n) * 100 / N_sents)

    parses = []

    decode_beats_proposal = 0
    decode_beats_gold = 0
    decode_beats_gold_and_proposal = 0
    # just the samples

    best_from_proposal = [
        max((lstm_score, parse) for (_, _, parse), lstm_score in zip(ipp, scores))
        for ipp, scores in zip(rnng_indices_discrim_scores_and_parses, rnng_lstm_scores)
    ]

    if args.output_reordered_decodes_prefix:
        decode_reordered_output_file = open(args.output_reordered_decodes_prefix, 'w')
    else:
        decode_reordered_output_file = None

    settings_with_best_decode = []

    reordered_decode_instances = []
    with open(best_proposal_fname, 'w') as f_proposal,    open(best_proposal_gold_fname, 'w') as f_proposal_gold,    open(best_proposal_decode_fname, 'w') as f_proposal_decode,    open(best_proposal_gold_decode_fname, 'w') as f_proposal_gold_decode,    open(only_decode_fname, 'w') as f_decode, open(only_gold_fname, 'w') as f_gold:
        for i, (best_proposal_score, best_proposal) in enumerate(best_from_proposal):
            best_proposal = best_proposal.replace("*HASH*", '#')
            gold_parse = rnng_gold_trees[i]
            decode_instances_for_this = [dis[rnng_indices_to_lstm_indices[i]]
                                         for dis in lstm_decode_indices_by_setting]

            reordered_decode_instances.append(decode_instances_for_this[0])

            # note: the tags in the decode_instance.gold_ptb are the actual gold
            # tags, not the ones predicted by the POS tagger, b/c of the data
            # fed to the LSTM
            gold_tags, gold_tokens, _ = get_tags_tokens_lowercase(gold_parse)
            for di in decode_instances_for_this:
                assert(gold_tokens == get_tags_tokens_lowercase(di.gold_ptb)[1])

                lstm_pred_tags, lstm_pred_tokens, _ = get_tags_tokens_lowercase(di.pred_ptb)

                assert(lstm_pred_tokens == gold_tokens)
                assert(len(lstm_pred_tags) == len(lstm_pred_tokens))
                assert(len(gold_tags) == len(gold_tokens))

                if decode_reordered_output_file is not None:
                    output_string = di.pred_ptb
                    # have to do this in two stages in case there are tokens with
                    # different overlapping tags
                    for ix, (lstm_pred_tag, lstm_pred_token) in enumerate(zip(lstm_pred_tags, lstm_pred_tokens)):
                        to_rep = '(' + lstm_pred_tag + ' ' + lstm_pred_token + ')'
                        assert(to_rep in output_string)
                        output_string = output_string.replace('(%s %s)' % (lstm_pred_tag, lstm_pred_token), '(XX %s)' % lstm_pred_token, 1)
                    for ix, (gold_tag, gold_token) in enumerate(zip(gold_tags, gold_tokens)):
                        to_rep = '(XX %s)' % gold_token
                        assert(to_rep in output_string)
                        output_string = output_string.replace(to_rep, '(%s %s)' % (gold_tag, gold_token), 1)
                    decode_reordered_output_file.write("%s\n" % output_string.replace('#', '*HASH*'))

            shuf_enum_dift = zip(args.lstm_search_prefixes, decode_instances_for_this)
            # break ties randomly
            random.shuffle(shuf_enum_dift)
            best_decode_setting, best_decode_instance = max(shuf_enum_dift, key=lambda (_, di): di.pred_score)
            settings_with_best_decode.append(best_decode_setting)
    #         print(i, best_proposal_score, decode_instance.pred_score, decode_instance.gold_score)
            f_proposal.write("%s\n" % best_proposal)

            f_gold.write("%s\n" % best_decode_instance.gold_ptb)
            f_decode.write("%s\n" % best_decode_instance.pred_ptb)

            if best_decode_instance.pred_score > best_proposal_score + 1e-3:
                decode_beats_proposal += 1

            if best_decode_instance.pred_score > best_decode_instance.gold_score + 1e-3:
                decode_beats_gold += 1

            if best_decode_instance.pred_score > best_decode_instance.gold_score + 1e-3 and best_decode_instance.pred_score > best_proposal_score + 1e-3:
                decode_beats_gold_and_proposal += 1

            f_proposal_decode.write("%s\n" % max([
                        (best_proposal_score, best_proposal),
                        (best_decode_instance.pred_score, best_decode_instance.pred_ptb)
                    ], key=lambda t: t[0])[1])

            f_proposal_gold.write("%s\n" % max([
                        (best_proposal_score, best_proposal),
                        (best_decode_instance.gold_score, best_decode_instance.gold_ptb)
                    ], key=lambda t: t[0])[1])

            f_proposal_gold_decode.write("%s\n" % max([
                        (best_proposal_score, best_proposal),
                        (best_decode_instance.pred_score, best_decode_instance.pred_ptb),
                        (best_decode_instance.gold_score, best_decode_instance.gold_ptb)
                    ], key=lambda t: t[0])[1])

    if args.output_reordered_decodes_prefix is not None:
        decode_reordered_output_file.close()
        with open(args.output_reordered_decodes_prefix + ".json", 'w') as f:
            json.dump([di._asdict() for di in reordered_decode_instances], f)

    if len(args.lstm_search_prefixes) > 1:
        output_to_stderr("settings with the best decodes:")
        output_to_stderr(Counter(settings_with_best_decode))

    output_to_stderr("decode beats proposal:\t" + percent_tuple_str(decode_beats_proposal))
    output_to_stderr("decode beats gold:\t" + percent_tuple_str(decode_beats_gold))
    output_to_stderr("decode beats gold and prop:\t" + percent_tuple_str(decode_beats_gold_and_proposal))
    output_to_stderr("")
    output_to_stderr("gold sanity check (R, P, F1, exact match):")
    output_to_stderr(evaluate.eval_b(rnng_gold_file, only_gold_fname))
    output_to_stderr("only decode (R, P, F1, exact match):")
    output_to_stderr(evaluate.eval_b(rnng_gold_file, only_decode_fname))
    output_to_stderr("rescore proposal (R, P, F1, exact match):")
    output_to_stderr(evaluate.eval_b(rnng_gold_file, best_proposal_fname))
    output_to_stderr("rescore proposal+gold (R, P, F1, exact match):")
    output_to_stderr(evaluate.eval_b(rnng_gold_file, best_proposal_gold_fname))
    output_to_stderr("rescore proposal+decode (R, P, F1, exact match):")
    output_to_stderr(evaluate.eval_b(rnng_gold_file, best_proposal_decode_fname))
    output_to_stderr("rescore proposal+decode+gold (R, P, F1, exact match):")
    output_to_stderr(evaluate.eval_b(rnng_gold_file, best_proposal_gold_decode_fname))
