import rnng_interpolate
import rnng_output_to_nbest
import rnng_threeway_interpolate
import decode_analysis
import json

import sys

from ptb_reader import get_tags_tokens_lowercase

import pandas
import evaluate

def output_to_stderr(line):
    sys.stderr.write("%s\n" % str(line))

if __name__ == "__main__":
    rnng_gold_file = '../rnng/corpora/22.auto.clean'
    rnng_samples_file = 'dyer_beam/dev_pos_embeddings_beam=100.ptb_samples'
    #rnng_lstm_score_file = 'dyer_beam/dev_pos_embeddings_beam=100.ptb_samples.likelihoods'
    rnng_gen_score_file = 'dyer_beam/dev_pos_embeddings_beam=100.samples.rnng-gen-likelihoods'

    # section 22, files in LSTM order
    lstm_stderr_file = sys.argv[1]
    lstm_decode_file = sys.argv[2]
    lstm_gold_file = sys.argv[3]
    rnng_lstm_score_file = sys.argv[4]
    if len(sys.argv) > 5:
        decode_reordered_output_fname = sys.argv[5]
        decode_instance_reordered_output_fname = sys.argv[6]
    else:
        decode_reordered_output_fname = None
        decode_instance_reordered_output_fname = None


    # lstm_gold_file = 'wsj/dev_22.txt.stripped'
    # lstm_decode_file = 'expts/beam_size=50-at_word=5.decode_all'
    # lstm_stderr_file = 'expts/beam_size=50-at_word=5.stderr_all'

    # RNNG order

    with open(rnng_gold_file) as f:
        rnng_gold_tokens = [get_tags_tokens_lowercase(line)[1] for line in f]
    with open(rnng_gold_file) as f:
        rnng_gold_trees = [line.strip() for line in f]

    with open(rnng_samples_file) as f:
        rnng_indices_discrim_scores_and_parses = list(rnng_output_to_nbest.parse_rnng_file(f))

    sample_lens = [len(l) for l in rnng_indices_discrim_scores_and_parses]

    with open(rnng_lstm_score_file) as f:
        rnng_lstm_scores = list(rnng_interpolate.parse_likelihood_file(f))

    with open(rnng_gen_score_file) as f:
        rnng_gen_scores = list(rnng_threeway_interpolate.parse_rnng_gen_score_file(f, sample_lens))

    # LSTM order

    with open(lstm_gold_file) as f:
        lstm_gold_tokens = [get_tags_tokens_lowercase(line)[1] for line in f]
    with open(lstm_gold_file) as f:
        lstm_gold_trees = [line.strip() for line in f]

    lstm_decode_instances = decode_analysis.parse_decode_output_files(lstm_gold_trees, lstm_decode_file, lstm_stderr_file)


    rnng_indices_to_lstm_indices = [
        lstm_gold_tokens.index(rnng_gt) for rnng_gt in rnng_gold_tokens
    ]

    best_proposal_fname = '/tmp/best_from_proposal.out'
    best_proposal_gold_fname = '/tmp/best_from_proposal_and_gold.out'
    best_proposal_decode_fname = '/tmp/best_from_proposal_and_decode.out'
    best_proposal_gold_decode_fname = '/tmp/best_from_proposal_gold_and_decode.out'

    only_decode_fname = '/tmp/only_decode.out'

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

    if decode_reordered_output_fname:
        decode_reordered_output_file = open(decode_reordered_output_fname, 'w')
    else:
        decode_reordered_output_file = None

    reordered_decode_instances = []
    with open(best_proposal_fname, 'w') as f_proposal,    open(best_proposal_gold_fname, 'w') as f_proposal_gold,    open(best_proposal_decode_fname, 'w') as f_proposal_decode,    open(best_proposal_gold_decode_fname, 'w') as f_proposal_gold_decode,    open(only_decode_fname, 'w') as f_decode:
        for i, (best_proposal_score, best_proposal) in enumerate(best_from_proposal):
            best_proposal = best_proposal.replace("*HASH*", '#')
            gold_parse = rnng_gold_trees[i]
            decode_instance = lstm_decode_instances[rnng_indices_to_lstm_indices[i]]
            reordered_decode_instances.append(decode_instance)

            # note: the tags in the decode_instance.gold_ptb are the actual gold
            # tags, not the ones predicted by the POS tagger, b/c of the data
            # fed to the LSTM
            gold_tags, gold_tokens, _ = get_tags_tokens_lowercase(gold_parse)
            assert(gold_tokens == get_tags_tokens_lowercase(decode_instance.gold_ptb)[1])

            lstm_pred_tags, lstm_pred_tokens, _ = get_tags_tokens_lowercase(decode_instance.pred_ptb)

            assert(lstm_pred_tokens == gold_tokens)
            assert(len(lstm_pred_tags) == len(lstm_pred_tokens))
            assert(len(gold_tags) == len(gold_tokens))

            if decode_reordered_output_file is not None:
                output_string = decode_instance.pred_ptb
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
    #         print(i, best_proposal_score, decode_instance.pred_score, decode_instance.gold_score)
            f_proposal.write("%s\n" % best_proposal)

            f_decode.write("%s\n" % decode_instance.gold_ptb)

            if decode_instance.pred_score > best_proposal_score + 1e-3:
                decode_beats_proposal += 1

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

    if decode_reordered_output_file is not None:
        decode_reordered_output_file.close()

    if decode_instance_reordered_output_fname:
        with open(decode_instance_reordered_output_fname, 'w') as f:
            json.dump([di._asdict() for di in reordered_decode_instances], f)

    output_to_stderr("decode beats proposal:\t" + percent_tuple_str(decode_beats_proposal))
    output_to_stderr("decode beats gold:\t" + percent_tuple_str(decode_beats_gold))
    output_to_stderr("decode beats gold and prop:\t" + percent_tuple_str(decode_beats_gold_and_proposal))
    output_to_stderr("")
    output_to_stderr("gold sanity check (R, P, F1, exact match):")
    output_to_stderr(evaluate.eval_b(rnng_gold_file, only_decode_fname))
    output_to_stderr("rescore proposal (R, P, F1, exact match):")
    output_to_stderr(evaluate.eval_b(rnng_gold_file, best_proposal_fname))
    output_to_stderr("rescore proposal+gold (R, P, F1, exact match):")
    output_to_stderr(evaluate.eval_b(rnng_gold_file, best_proposal_gold_fname))
    output_to_stderr("rescore proposal+decode (R, P, F1, exact match):")
    output_to_stderr(evaluate.eval_b(rnng_gold_file, best_proposal_decode_fname))
    output_to_stderr("rescore proposal+decode+gold (R, P, F1, exact match):")
    output_to_stderr(evaluate.eval_b(rnng_gold_file, best_proposal_gold_decode_fname))
