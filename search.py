from __future__ import print_function
from collections import namedtuple
import numpy as np
import heapq
import sys
import reader
import tensorflow as tf
import pickle
from utils import PTBModel
import utils
import score
import copy
import search
import time
import ptb_reader

__author__ = 'dfried'

OPEN_NT = 0
SHIFT = 1
CLOSE_NT = 2
FINAL_EOS = 3

ConsCell = namedtuple('ConsCell', 'car, cdr')

BeamState = namedtuple('BeamState', 'prev_beam_state, lstm_state, prev_word_index, open_nt_stack, open_nt_count, cons_nt_count, action_count, prev_action_type, prev_action_index, term_count, score')

def make_matching_brackets(word_to_id):
    pairs = {}
    for key in word_to_id:
        if key.startswith("(") or key.startswith(")"):
            nt = key[1:]
            pairs[nt] = (word_to_id["("+nt], word_to_id[")"+nt])
    return pairs

def backchain_beam_state(final_beam_state):
    action_indices = []
    word_indices = []
    beam_state = final_beam_state
    while beam_state is not None:
        action_indices.append(beam_state.prev_action_index)
        if beam_state.prev_word_index is not None:
            word_indices.append(beam_state.prev_word_index)
        beam_state = beam_state.prev_beam_state
    return list(reversed(action_indices)), list(reversed(word_indices))

class BeamSearch(object):
    def __init__(self, session, model, word_to_id, max_open_nts=100, max_cons_nts=8):
        self.session = session
        self.model = model
        self.word_to_id = word_to_id
        self.id_to_word = {id_:word for (word, id_) in word_to_id.items()}
        self.matching_brackets = make_matching_brackets(word_to_id)

        # [S, NT, ...] not necessarily in that order
        self.nts_by_index = []
        # [ vocab index of "(S", vocab index of "(NT", ...]
        self.open_indices = []
        # [ vocab index of ")S", vocab index of ")NT", ...]
        self.close_indices = []
        for nt, (open_ix, close_ix) in sorted(self.matching_brackets.items()):
            self.nts_by_index.append(nt)
            self.open_indices.append(open_ix)
            self.close_indices.append(close_ix)

        self.eos_index = word_to_id['<eos>']
        self.max_open_nts = max_open_nts
        self.max_cons_nts = max_cons_nts

    def score_actions(self, beam_state):
        # update hidden state bsed on the pre-chosen prev_action_index and get a distribution over actions
        feed_dict = {
            self.model.input_data: np.array([[beam_state.prev_action_index]])
        }
        for (state_c, state_h), (c, h) in zip(beam_state.lstm_state, self.model.initial_state):
            feed_dict[c] = state_c
            feed_dict[h] = state_h
        fetches = [self.model.log_probs]
        for c, h in self.model.final_state:
            fetches.append(c)
            fetches.append(h)
        res = self.session.run(fetches, feed_dict)
        log_probs = res[0]
        lstm_state_flat = res[1:] # [c1, h1, c2, h2...]
        lstm_state = [lstm_state_flat[i:i+2] for i in range(0, len(lstm_state_flat), 2)]
        new_state = beam_state._replace(lstm_state=lstm_state)
        return log_probs, new_state

    def choose_action(self, beam_state, action_type, nt_index, action_index, action_score, len_terms):
        # set this to be the word generated if we are shifting
        new_prev_word_index = None
        new_open_nt_stack = beam_state.open_nt_stack
        new_open_nt_count = beam_state.open_nt_count
        new_cons_nt_count = beam_state.cons_nt_count
        new_term_count = beam_state.term_count

        if action_type == OPEN_NT:
            new_open_nt_stack = ConsCell(nt_index, beam_state.open_nt_stack)
            new_open_nt_count += 1
            new_cons_nt_count += 1
        elif action_type == SHIFT:
            assert(nt_index is None)
            new_prev_word_index = action_index
            new_term_count += 1
            new_cons_nt_count = 0
        elif action_type == CLOSE_NT:
            assert(beam_state.open_nt_count > 0) # is something to close
            if beam_state.open_nt_count == 1: # check to make sure there are no remaining words
                assert(beam_state.term_count == len_terms)
            assert(nt_index == beam_state.open_nt_stack.car)  # make sure we match the opening terminal
            new_open_nt_stack = beam_state.open_nt_stack.cdr
            new_open_nt_count -= 1
            new_cons_nt_count = 0
            assert(new_open_nt_count >= 0)
            assert((new_open_nt_stack is None) == (new_open_nt_count == 0))
        else:
            assert(action_type == FINAL_EOS)
            assert(action_index == self.eos_index)
            assert(new_open_nt_stack is None)
            assert(new_open_nt_count == 0)

        new_state = beam_state._replace(
            prev_beam_state=beam_state,
            # lstm_state stays the same, is updated by score_actions
            prev_word_index=new_prev_word_index,
            open_nt_stack=new_open_nt_stack,
            open_nt_count=new_open_nt_count,
            cons_nt_count=new_cons_nt_count,
            action_count=beam_state.action_count + 1,
            term_count=new_term_count,
            prev_action_type=action_type,
            prev_action_index=action_index,
            score=beam_state.score + action_score
        )
        return new_state

    def valid_actions(self, beam_state, term_ids, len_terms):
        # return list of valid (action_type, nt_index, action_index) for the current beam_state
        remaining_terms = len_terms - beam_state.term_count
        if beam_state.prev_action_type is None:
            # can only open a new NT
            return [(OPEN_NT, nt_index, action_index) for nt_index, action_index in enumerate(self.open_indices)]

        actions = []
        # check open actions: must be below the limit and have some term remaining
        if beam_state.cons_nt_count < self.max_cons_nts and beam_state.open_nt_count < self.max_open_nts and remaining_terms > 0:
            actions.extend((OPEN_NT, nt_index, action_index) for nt_index, action_index in enumerate(self.open_indices))
        # check shift action: can shift if there are any words remaining
        if remaining_terms > 0:
            actions.append((SHIFT, None, term_ids[beam_state.term_count]))
        # check close action: can't close immediately after an open, can't close a top level paren unless we've exhausted all remaining terms
        if beam_state.open_nt_count > 0 and beam_state.prev_action_type != OPEN_NT and (beam_state.open_nt_count > 1 or remaining_terms == 0):
            actions.append((CLOSE_NT, beam_state.open_nt_stack.car, self.close_indices[beam_state.open_nt_stack.car]))
        if beam_state.open_nt_count == 0:
            assert(remaining_terms == 0)
            actions.append((FINAL_EOS, None, self.eos_index))
        assert(actions)
        return actions

    def beam_search(self, term_ids, beam_size):
        initial_lstm_state = []
        for c, h in self.model.initial_state:
            initial_lstm_state.append((c.eval(), h.eval()))

        assert(term_ids[0] != self.eos_index and term_ids[-1] != self.eos_index)

        len_terms = len(term_ids)

        initial_beam_state = BeamState(prev_beam_state=None,
                                       lstm_state=initial_lstm_state,
                                       prev_word_index=None,
                                       open_nt_stack=None,
                                       open_nt_count=0,
                                       cons_nt_count=0,
                                       action_count=0,
                                       term_count=0,
                                       prev_action_type=None,
                                       prev_action_index=self.eos_index,
                                       score=0)

        # list of beam states
        beam = [initial_beam_state]
        # list of beam states
        completed = []

        def successor_scoring_fn(successor_tuple):
            (action_score, _, _, _, old_beam_state) = successor_tuple
            return action_score + old_beam_state.score

        while len(completed) < beam_size and beam:
            # list of (action_score, action_type, nt_index, action_index, old beam_state)
            successors = []

            while beam:
                # build successors for each item in the beam
                current_beam_state = beam.pop()
                action_scores, updated_beam_state = self.score_actions(current_beam_state)

                for t in self.valid_actions(updated_beam_state, term_ids, len_terms):
                    (action_type, nt_index, action_index) = t
                    action_score = action_scores[0, action_index] # 1 x |vocab|
                    successors.append((action_score, action_type, nt_index, action_index, updated_beam_state))
                # end beam successors

            # get the beam_size best successors, put them back on the beam (or completed if they've produced EOS)
            for (action_score, action_type, nt_index, action_index, old_beam_state) in heapq.nlargest(beam_size, successors, key=successor_scoring_fn):
                new_beam_state = self.choose_action(old_beam_state, action_type, nt_index, action_index, action_score, len_terms)
                if new_beam_state.prev_action_type == FINAL_EOS:
                    assert(new_beam_state.open_nt_count == 0)
                    assert(new_beam_state.term_count == len_terms)
                    completed.append(new_beam_state)
                else:
                    beam.append(new_beam_state)

            if beam:
                sys.stderr.write("\r%s" % beam[0].action_count)

        sys.stderr.write("\r")

        best_state = max(completed, key=lambda beam_state: beam_state.score)

        best_action_indices, best_word_indices = backchain_beam_state(best_state)
        assert(best_word_indices == term_ids)
        return best_action_indices, best_state.score

    def beam_search_within_word(self, term_ids, beam_size, beam_size_at_word):
        initial_lstm_state = []
        for c, h in self.model.initial_state:
            initial_lstm_state.append((c.eval(), h.eval()))

        assert(term_ids[0] != self.eos_index and term_ids[-1] != self.eos_index)

        len_terms = len(term_ids)

        initial_beam_state = BeamState(prev_beam_state=None,
                                       lstm_state=initial_lstm_state,
                                       prev_word_index=None,
                                       open_nt_stack=None,
                                       open_nt_count=0,
                                       cons_nt_count=0,
                                       action_count=0,
                                       term_count=0,
                                       prev_action_type=None,
                                       prev_action_index=self.eos_index,
                                       score=0)

        # list of beam states
        beam = [initial_beam_state]
        # list of beam states
        completed = []

        def successor_scoring_fn(successor_tuple):
            (action_score, _, _, _, old_beam_state) = successor_tuple
            return action_score + old_beam_state.score

        for current_word_count in range(len_terms):
            sys.stderr.write("\r%d / %d" % (current_word_count + 1,  len_terms))
            completed = []

            # todo: check against beam_size_at_word
            while len(completed) < beam_size and beam:
                # list of (action_score, action_type, nt_index, action_index, old beam_state)
                successors = []

                while beam:
                    # build successors for each item in the beam
                    current_beam_state = beam.pop()
                    action_scores, updated_beam_state = self.score_actions(current_beam_state)

                    for t in self.valid_actions(updated_beam_state, term_ids, len_terms):
                        (action_type, nt_index, action_index) = t
                        action_score = action_scores[0, action_index] # 1 x |vocab|
                        successors.append((action_score, action_type, nt_index, action_index, updated_beam_state))
                        # end beam successors

                # get the beam_size best successors, put them back on the beam (or completed if they've produced EOS)
                for (action_score, action_type, nt_index, action_index, old_beam_state) in heapq.nlargest(beam_size, successors, key=successor_scoring_fn):
                    new_beam_state = self.choose_action(old_beam_state, action_type, nt_index, action_index, action_score, len_terms)
                    if current_word_count == len_terms - 1: # completion is producing EOS
                        if new_beam_state.prev_action_type == FINAL_EOS:
                            assert(new_beam_state.open_nt_count == 0)
                            assert(new_beam_state.term_count == len_terms)
                            completed.append(new_beam_state)
                        else:
                            beam.append(new_beam_state)
                    else:
                        if new_beam_state.prev_action_type == SHIFT:
                            completed.append(new_beam_state)
                        else:
                            beam.append(new_beam_state)


            beam = heapq.nlargest(beam_size_at_word, completed, key=lambda bi: bi.score)
            sys.stderr.write(" %s" % ' '.join(self.id_to_word[id_] for id_ in backchain_beam_state(max(beam, key=lambda bi: bi.score))[0]))

        sys.stderr.write("\r")

        best_state = max(completed, key=lambda beam_state: beam_state.score)

        best_action_indices, best_word_indices = backchain_beam_state(best_state)
        assert(best_word_indices == term_ids)
        return best_action_indices, best_state.score

def get_words(id_to_word, action_indices):
    return [id_ for id_ in action_indices if not (id_to_word[id_] == '<eos>' or id_to_word[id_].startswith('(') or id_to_word[id_].startswith(')'))]

def display_parse(id_to_word, action_indices):
    return ' '.join([id_to_word[id_] for id_ in action_indices])

def partition(lst, n):
    # http://stackoverflow.com/questions/2659900/python-slicing-a-list-into-n-nearly-equal-length-partitions
    q, r = divmod(len(lst), n)
    indices = [q*i + min(i, r) for i in xrange(n+1)]
    return [lst[indices[i]:indices[i+1]] for i in xrange(n)]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="wsj")
    parser.add_argument("--model_path", default="models/wsj/model")
    parser.add_argument("--train_path", default="wsj/train_02-21.txt.traversed")
    parser.add_argument("--valid_path", default="wsj/dev_22.txt.traversed")
    parser.add_argument("--valid_nbest_path", default="wsj/dev_22.txt.nbest")
    parser.add_argument("--valid_nbest_traversed_path", default="wsj/dev_22.txt.nbest.traversed")
    parser.add_argument("--gold_dev_stripped_path", default="wsj/dev_22.txt.stripped")
    parser.add_argument("--beam_size", type=int, default=50)
    parser.add_argument("--beam_size_at_word", type=int, default=5)
    parser.add_argument("--beam_within_word", action='store_true')
    parser.add_argument("--decode_file")
    parser.add_argument("--block_count", type=int)
    parser.add_argument("--block_num", type=int)

    args=parser.parse_args()

    config = pickle.load(open(args.model_path + '.config', 'rb'))
    config.batch_size = 10

    train_data, valid_data, _, word_to_id = reader.ptb_raw_data(args.data_path, train_path=args.train_path, valid_path=args.valid_path, valid_nbest_path=args.valid_nbest_traversed_path)

    id_to_word = {v:k for k,v in word_to_id.items()}

    eos_index = word_to_id['<eos>']
    num_sents = len([id_ for id_ in valid_data if id_ == eos_index])

    if args.block_count:
        assert(args.block_num is not None)
        this_block = partition(range(num_sents), args.block_count)[args.block_num]
        sys.stderr.write("will only decode sentences in %d .. %d\n" % (this_block[0], this_block[-1]))
    else:
        this_block = None

    if args.decode_file:
        fname = args.decode_file
        if args.block_num:
            fname = fname + "_block-%d" % args.block_num
        f_decode = open(fname, 'w', buffering=0)
    else:
        f_decode = sys.stdout

    with tf.Graph().as_default(), tf.Session() as session:
        small_config = copy.copy(config)
        small_config.batch_size = 1
        small_config.num_steps = 1
        initializer = tf.random_uniform_initializer(-small_config.init_scale,
                                                    small_config.init_scale)

        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = PTBModel(is_training=False, config=small_config)
        saver = tf.train.Saver()
        saver.restore(session, args.model_path)

        bs = search.BeamSearch(session, m, word_to_id)

        with open(args.gold_dev_stripped_path) as f_dev:
            for i, (gold_ptb_line, gold_action_indices) in enumerate(zip(f_dev, reader.ptb_iterator_single_sentence(valid_data, eos_index))):
                if this_block is not None and i not in this_block:
                    continue
                word_indices = get_words(id_to_word, gold_action_indices)
                start_time = time.time()
                if args.beam_within_word:
                    pred_action_indices, pred_score = bs.beam_search_within_word(word_indices, args.beam_size, args.beam_size_at_word)
                else:
                    pred_action_indices, pred_score = bs.beam_search(word_indices, args.beam_size)
                end_time = time.time()
                pred_rescore = -score.score_single_tree(session, m, np.array(pred_action_indices))
                if abs(pred_rescore - pred_score) < 1e-3:
                    sys.stderr.write("WARNING: score mismatch, beam %s, rescore %s" % (pred_score, pred_rescore))
                gold_ptb_tags, gold_ptb_tokens, _  = ptb_reader.get_tags_tokens_lowercase(gold_ptb_line)
                sys.stderr.write("sentence %s\n" % i)
                sys.stderr.write("gold sent:\t%s\n" % display_parse(id_to_word, word_indices))
                sys.stderr.write("pred sent:\t%s\n" % display_parse(id_to_word, get_words(id_to_word, pred_action_indices)))
                sys.stderr.write("gold:\t%s\n" % display_parse(id_to_word, gold_action_indices))
                sys.stderr.write("pred:\t%s\n" % display_parse(id_to_word, pred_action_indices))
                sys.stderr.write("gold score:\t%s\n" % -score.score_single_tree(session, m, np.array(gold_action_indices)))
                sys.stderr.write("pred score:\t%s\n" % pred_score)
                sys.stderr.write("match?:\t%s\n" % (gold_action_indices == pred_action_indices))
                sys.stderr.write("%0.2f seconds\n" % (end_time - start_time))
                sys.stderr.write("\n")
                f_decode.write("%s\n" % ' '.join(utils.convert_to_ptb_format(id_to_word, pred_action_indices, gold_tags=gold_ptb_tags, gold_tokens=gold_ptb_tokens)))

    if args.decode_file:
        f_decode.close()
