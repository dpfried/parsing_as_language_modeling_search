from __future__ import absolute_import, division, print_function
from utils import PTBModel, MediumConfig, _build_vocab

import time
import pickle
import tensorflow as tf

from score import score_all_trees

from rnng_output_to_nbest import parse_rnng_file

import reader

# flags = tf.flags
logging = tf.logging

def rerank(train_traversed_path, candidate_path, model_path, output_path, sent_limit=None, likelihood_file=None, best_file=None):
    # candidate path: file in ix ||| candidate score ||| parse format
    config = pickle.load(open(model_path + '.config', 'rb'))
    config.batch_size = 10

    with open(candidate_path) as f:
        candidates_by_sent = list(parse_rnng_file(f))

    if sent_limit is not None:
        candidates_by_sent = candidates_by_sent[:sent_limit]

    parses_by_sent = [
        ["(S1 %s)" % parse.replace("*HASH*", "#") for (ix, score, parse) in candidates]
        for candidates in candidates_by_sent
    ]

    word_to_id = _build_vocab(train_traversed_path)
    test_nbest_data = reader.ptb_list_to_word_ids(parses_by_sent,
                                                  word_to_id,
                                                  remove_duplicates=False,
                                                  sent_limit=None)

    assert(len(test_nbest_data['trees']) == len(parses_by_sent))
    assert(all(len(x) == len(y) for x, y in zip(parses_by_sent, test_nbest_data['trees'])))

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = PTBModel(is_training=False, config=config)

        saver = tf.train.Saver()
        saver.restore(session, model_path)
        losses_by_sent = score_all_trees(session, m, test_nbest_data, tf.no_op(), word_to_id['<eos>'], likelihood_file=likelihood_file, output_nbest=best_file)

    assert(len(losses_by_sent) == len(candidates_by_sent))
    with open(output_path, 'w') as f:
        for sent_ix, (candidates, losses) in enumerate(zip(candidates_by_sent, losses_by_sent)):
            assert(len(candidates) == len(losses))
            for (ix, old_score, parse), loss in zip(candidates, losses):
                f.write("%s ||| %s ||| %s\n" % (ix, -loss, parse))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("train_traversed_path", help="e.g. wsj/train_02-21.txt.traversed")
    parser.add_argument("model_path")
    parser.add_argument("candidate_path")
    parser.add_argument("output_path")
    parser.add_argument("--sent_limit", type=int)
    parser.add_argument("--likelihood_file", help="additionally output scores to this file")
    parser.add_argument("--best_file", help="output best parses to this file")
    args = parser.parse_args()

    rerank(args.train_traversed_path,
           args.candidate_path,
           args.model_path,
           args.output_path,
           args.sent_limit,
           args.likelihood_file,
           args.best_file)
