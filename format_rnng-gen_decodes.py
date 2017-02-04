import rnng_output_to_nbest
import rnng_decode_analysis
import numpy as np

import sys

from ptb_reader import get_tags_tokens_lowercase, remove_dev_unk


def output_to_stderr(line):
    sys.stderr.write("%s\n" % str(line))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("rnng_gold_file_stripped")
    parser.add_argument("rnng_search_prefix", help="list of prefixes for files ending in stdout_block--all, decode_block--all, and _beam_block--all")
    parser.add_argument("output_file")
    parser.add_argument("--max_num_candidates", type=int)

    args = parser.parse_args()

    with open(args.rnng_gold_file_stripped) as f:
        rnng_gold_tokens = [get_tags_tokens_lowercase(line)[1] for line in f]
    with open(args.rnng_gold_file_stripped) as f:
        rnng_gold_trees = [line.strip() for line in f]

    decode_file = args.rnng_search_prefix + ".decode_block--all"
    stdout_file = args.rnng_search_prefix + ".stdout_block--all"
    beam_file = args.rnng_search_prefix + ".beam_block--all"

    rnng_decode_instances = rnng_decode_analysis.parse_decode_output_files(args.rnng_gold_file_stripped, decode_file, stdout_file)

    N_sents = len(rnng_decode_instances)
    print("%d sents" % N_sents)

    all_candidates = []

    candidate_count = []

    with open(beam_file) as f:
        for rnng_ix, candidates in enumerate(rnng_output_to_nbest.parse_rnng_file(f)):
            sys.stderr.write("\r%d / %d" % (rnng_ix, N_sents))
            if args.max_num_candidates and len(candidates) > args.max_num_candidates:
                candidates = candidates[:args.max_num_candidates]

            assert(all(ix == rnng_ix for (ix, _, _) in candidates))
            decode_inst = rnng_decode_instances[rnng_ix]

            gold_tree = rnng_gold_trees[rnng_ix]
            gold_tags, gold_tokens, _ = get_tags_tokens_lowercase(gold_tree)

            inst_gold_tags, inst_gold_tokens, _ = get_tags_tokens_lowercase(decode_inst.gold_ptb)
            inst_pred_tags, inst_pred_tokens, _ = get_tags_tokens_lowercase(decode_inst.pred_ptb)

            cand_tags, cand_tokens, _ = get_tags_tokens_lowercase(candidates[0][2])

            assert(gold_tags == inst_gold_tags)
            assert(gold_tokens == inst_gold_tokens)

            assert(gold_tags == inst_pred_tags)
            assert(gold_tokens == inst_pred_tokens)

            assert(abs(decode_inst.pred_score - candidates[0][1]) < 1e-3)

            def process(candidate_parse):
                return remove_dev_unk(gold_tree, candidate_parse)

            candidate_count.append(len(candidates))

            all_candidates.append([(ix, score, process(parse)) for
                                   (ix, score, parse) in candidates])

    sys.stderr.write("\n")

    print(len(candidate_count) == N_sents)
    # TODO min beam, max beam
    print("min candidate count: ", min(candidate_count))
    print("max candidate count: ", max(candidate_count))
    print("avg candidate count: ", np.mean(candidate_count))

    assert(set(cands[0][0] for cands in all_candidates) == set(range(N_sents)))

    with open(args.output_file, 'w') as f:
        for cands_by_sent in all_candidates:
            for (ix, score, parse) in cands_by_sent:
                f.write("%s ||| %s ||| %s\n" % (ix, score, parse))
