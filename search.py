__author__ = 'dfried'

from collections import namedtuple
import numpy as np

def make_matching_brackets(word_to_id):
  pairs = {}
  for key in word_to_id:
    if key.startswith("(") or key.endswith(")"):
      nt = key[1:]
      pairs[nt] = (word_to_id["("+nt], word_to_id[")"+nt])
  return pairs

OPEN_NT = 0
SHIFT = 1
CLOSE_NT = 2

ConsCell = namedtuple('ConsCell', 'car, cdr')

BeamState = namedtuple('BeamState', 'prev_beam_state, lstm_state, prev_word_index, open_nt_stack, open_nt_count, cons_nt_count, action_count, prev_action, term_count, score')

class BeamSearch(object):
  def __init__(self, session, model, word_to_id, max_open_nts=100, max_cons_nts=8):
    self.session = session
    self.model = model
    self.word_to_id = word_to_id
    self.matching_brackets = make_matching_brackets(word_to_id)
    self.open_indices = [v[0] for v in self.matching_brackets.values()]
    self.close_indices = [v[1] for v in self.matching_brackets.values()]
    self.eos_index = word_to_id['<eos>']
    self.max_open_nts = max_open_nts
    self.max_cons_nts = max_cons_nts

  def valid_actions(self, beam_state, term_ids, len_terms):
    remaining_terms = len_terms - beam_state.term_count
    if beam_state.prev_action is None:
      # can only open a new NT
      return self.open_indices

    actions = []
    # check open actions: must be below the limit and have some term remaining
    if beam_state.cons_nt_count < self.max_cons_nts and beam_state.open_nt_count < self.max_open_nts and remaining_terms > 0:
      actions.extend(self.open_indices)
    # check shift action: can shift if there are any words remaining
    if remaining_terms > 0:
      actions.append(term_ids[beam_state.term_count])
    # check close action: can't close immediately after an open, can't close a top level paren unless we've exhausted all remaining terms
    if beam_state.prev_action != OPEN_NT and (beam_state.open_nt_count > 1 or remaining_terms == 0):
        actions.append(self.close_indices[beam_state.open_nt_stack.car])
    return actions

  def beam_search(self, term_ids, beam_size):
    initial_lstm_state = []
    for c, h in self.model.initial_state:
      initial_lstm_state.append((c.eval(), h.eval()))

    assert(term_ids[0] != self.eos_index and term_ids[-1] != self.eos_index)

    len_terms = len(term_ids)

    initial_beam_state = BeamState(prev_beam_state=None,
                                   lstm_state=initial_lstm_state,
                                   prev_word_index=self.eos_index,
                                   open_nt_stack=None,
                                   open_nt_count=0,
                                   cons_nt_count=0,
                                   action_count=0,
                                   term_count=0,
                                   prev_action=None,
                                   score=0)

    beam = []
    completed = []

    while len(completed < beam_size and beam):
      successors = []

      while beam:
        current_beam_item = beam.pop()

        feed_dict = {
          self.model.input_data: np.array([[current_beam_item.prev_word_index]])
        }
        for (state_c, state_h), (c, h) in zip(current_beam_item.lstm_state, self.model.initial_state):
          feed_dict[c] = state_c
          feed_dict[h] = state_h
        log_probs = self.session.run(self.model.log_probs, feed_dict)

        for actions in self.valid_actions(current_beam_item, term_ids, len_terms):


    for i in range(len(x) - 1):
      input_arr = x[i:i+1][:,None]
      target_arr = x[i+1:i+2][:,None]
      fetches = [self.model.log_probs]
      for c, h in self.model.final_state:
        fetches.append(c)
        fetches.append(h)
      feed_dict = {
        self.model.input_data: input_arr
      }
      for (state_c, state_h), (c, h) in zip(state, self.model.initial_state):
        feed_dict[c] = state_c
        feed_dict[h] = state_h
      res = self.session.run(fetches, feed_dict)
      cost = res[0]
      costs.append(cost[0])
      state_flat = res[1:] # [c1, h1, c2, h2...]
      state = [state_flat[i:i+2] for i in range(0, len(state_flat), 2)]
      total_cost += np.sum(cost)
    print(costs)
    return total_cost
