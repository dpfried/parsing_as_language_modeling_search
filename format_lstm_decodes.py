import rnng_output_to_nbest
import decode_analysis
import re
import numpy as np

import sys

from ptb_reader import get_tags_tokens_lowercase


def output_to_stderr(line):
    sys.stderr.write("%s\n" % str(line))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("lstm_gold_file")
    parser.add_argument("rnng_gold_file_unstripped")
    parser.add_argument("lstm_search_prefix", help="list of prefixes for files ending in stderr_block--all, decode_block--all, and _beam_block--all")
    parser.add_argument("output_file")
    parser.add_argument("--max_num_candidates", type=int)

    args = parser.parse_args()

    rnng_gold_file_unstripped = args.rnng_gold_file_unstripped if args.rnng_gold_file_unstripped is not None else '../rnng/corpora/22.auto.clean'

    # RNNG order

    with open(rnng_gold_file_unstripped) as f:
        rnng_gold_tokens = [get_tags_tokens_lowercase(line)[1] for line in f]
    with open(rnng_gold_file_unstripped) as f:
        rnng_gold_trees = [line.strip() for line in f]

    # LSTM order

    with open(args.lstm_gold_file) as f:
        lstm_gold_tokens = [get_tags_tokens_lowercase(line)[1] for line in f]
    with open(args.lstm_gold_file) as f:
        lstm_gold_trees = [line.strip() for line in f]

    decode_file = args.lstm_search_prefix + ".decode_block--all"
    stderr_file = args.lstm_search_prefix + ".stderr_block--all"
    beam_file = args.lstm_search_prefix + ".beam_block--all"

    lstm_decode_instances = decode_analysis.parse_decode_output_files(args.lstm_gold_file, decode_file, stderr_file)

    N_sents = len(lstm_decode_instances)
    print("%d sents" % N_sents)

    def strip_root_and_tags(parse, root):
        assert(parse.startswith("(" + root + " "))
        parse = parse[len(root) + 3:-1]
        return re.sub("\(\S+ (\S+)\)", "(XX \\1)", parse)

    rnng_gold_trees_stripped = [strip_root_and_tags(parse, "TOP") for parse in rnng_gold_trees]
    lstm_gold_trees_stripped = [strip_root_and_tags(parse, "S1") for parse in lstm_gold_trees]

    # rnng_indices_to_lstm_indices = [
    #     lstm_gold_trees_stripped.index(rnng_gt) for rnng_gt in rnng_gold_trees_stripped
    # ]

    lstm_indices_to_rnng_indices = []
    for lstm_gt in lstm_gold_trees_stripped:
        start=0
        rnng_ix = None
        rnng_ix = rnng_gold_trees_stripped.index(lstm_gt, start)
        while rnng_ix in lstm_indices_to_rnng_indices:
            start+=1
            rnng_ix = rnng_gold_trees_stripped.index(lstm_gt, start)
        lstm_indices_to_rnng_indices.append(rnng_ix)

    # lstm_indices_to_rnng_indices = [
    #     rnng_gold_trees_stripped.index(lstm_gt) for lstm_gt in lstm_gold_trees_stripped
    # ]
    assert(len(set(lstm_indices_to_rnng_indices)) == len(lstm_indices_to_rnng_indices))

    all_candidates = []

    candidate_count = []

    with open(beam_file) as f:
        # these candidates are in lstm order (coming from an lstm beam search),
        # but are in rnng format
        for lstm_ix, candidates in enumerate(rnng_output_to_nbest.parse_rnng_file(f)):
            sys.stderr.write("\r%d / %d" % (lstm_ix, N_sents))
            if args.max_num_candidates and len(candidates) > args.max_num_candidates:
                candidates = candidates[:args.max_num_candidates]

            lstm_tags, lstm_tokens, _ = get_tags_tokens_lowercase(lstm_gold_trees[lstm_ix])

            assert(all(ix == lstm_ix for (ix, _, _) in candidates))
            decode_inst = lstm_decode_instances[lstm_ix]

            rnng_ix = lstm_indices_to_rnng_indices[lstm_ix]
            rnng_tags, rnng_tokens, _ = get_tags_tokens_lowercase(rnng_gold_trees[rnng_ix])

            assert(lstm_tokens == get_tags_tokens_lowercase(decode_inst.gold_ptb)[1])
            assert(lstm_tokens == get_tags_tokens_lowercase(decode_inst.pred_ptb)[1])
            assert(lstm_tokens == get_tags_tokens_lowercase(candidates[0][2])[1])
            assert(len(rnng_tags) == len(lstm_tags))
            assert(len(rnng_tokens) == len(lstm_tokens))

            assert(all(r.replace('*HASH*', '#') == l for (r, l) in zip(rnng_tokens, lstm_tokens)))

            assert(abs(decode_inst.pred_score - candidates[0][1]) < 1e-3)

            def process_parse(parse):
                for ix, (lstm_tag, lstm_token) in enumerate(zip(lstm_tags, lstm_tokens)):
                    to_rep = '(' + lstm_tag + ' ' + lstm_token + ')'
                    assert(to_rep in parse)
                    parse = parse.replace('(%s %s)' % (lstm_tag, lstm_token), '(XX %s)' % lstm_token, 1)
                for ix, (rnng_tag, rnng_token) in enumerate(zip(rnng_tags, rnng_tokens)):
                    to_rep = '(XX %s)' % rnng_token
                    assert(to_rep in parse)
                    parse = parse.replace(to_rep, '(%s %s)' % (rnng_tag, rnng_token), 1)
                return parse.replace('#', '*HASH')

            candidate_count.append(len(candidates))

            all_candidates.append([(rnng_ix, score, process_parse(parse)) for (_, score, parse) in candidates])

            # TODO: replace '#' -> '*HASH*'
    sys.stderr.write("\n")

    print(len(candidate_count) == N_sents)
    # TODO min beam, max beam
    print("min candidate count: ", min(candidate_count))
    print("max candidate count: ", max(candidate_count))
    print("avg candidate count: ", np.mean(candidate_count))

    assert(set(cands[0][0] for cands in all_candidates) == set(range(N_sents)))

    # sort by rnng index
    all_candidates = sorted(all_candidates, key=lambda candidates: candidates[0][0])

    with open(args.output_file, 'w') as f:
        for cands_by_sent in all_candidates:
            for (ix, score, parse) in cands_by_sent:
                f.write("%s ||| %s ||| %s\n" % (ix, score, parse))
