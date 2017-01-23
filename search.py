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

BeamState = namedtuple('BeamState', 'prev_beam_state, lstm_state, prev_word_index, open_nt_stack, open_nt_count, cons_nt_count, action_count, prev_action_type, prev_action_index, term_count, score')

class BeamSearch(object):
  def __init__(self, session, model, word_to_id, max_open_nts=100, max_cons_nts=8):
    self.session = session
    self.model = model
    self.word_to_id = word_to_id
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

  def valid_actions(self, beam_state, term_ids, len_terms):
    # return list of valid (action_type, nt_index, action_index) for the current beam_state
    remaining_terms = len_terms - beam_state.term_count
    if beam_state.prev_action_type is None:
      # can only open a new NT
      return self.open_indices

    actions = []
    # check open actions: must be below the limit and have some term remaining
    if beam_state.cons_nt_count < self.max_cons_nts and beam_state.open_nt_count < self.max_open_nts and remaining_terms > 0:
      actions.extend((OPEN_NT, nt_index, action_index) for nt_index, action_index in enumerate(self.open_indices))
    # check shift action: can shift if there are any words remaining
    if remaining_terms > 0:
      actions.append((SHIFT, None, term_ids[beam_state.term_count]))
    # check close action: can't close immediately after an open, can't close a top level paren unless we've exhausted all remaining terms
    if beam_state.prev_action_type != OPEN_NT and (beam_state.open_nt_count > 1 or remaining_terms == 0):
        actions.append((CLOSE_NT, beam_state.open_nt_stack.car, self.close_indices[beam_state.open_nt_stack.car]))
    return actions

  def beam_search(self, term_ids, beam_size):
    initial_lstm_state = []
    for c, h in self.model.initial_state:
      initial_lstm_state.append((c.eval(), h.eval()))

    assert(term_ids[0] != self.eos_index and term_ids[-1] != self.eos_index)

    len_terms = len(term_ids)

    def score_actions(beam_state):
      feed_dict = {
        self.model.input_data: np.array([[current_beam_item.prev_action_index]])
      }
      for (state_c, state_h), (c, h) in zip(current_beam_item.lstm_state, self.model.initial_state):
        feed_dict[c] = state_c
        feed_dict[h] = state_h
      fetches = [self.model.log_probs]
      for c, h in self.model.final_state:
        fetches.append(c)
      res = self.session.run(fetches, feed_dict)
      log_probs = res[0]
      lstm_state_flat = res[1:] # [c1, h1, c2, h2...]
      lstm_state = [state_flat[i:i+2] for i in range(0, len(lstm_state_flat), 2)]
      new_state = beam_state._replace(lstm_state=lstm_state)
      return log_probs, new_state

    def choose_action(beam_state, action_type, nt_index, action_index, action_score):
      new_prev_word_index = beam_state.prev_word_index
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
      else:
        assert(action_type == CLOSE_NT)
        assert(beam_state.open_nt_count > 0) # is something to close
        if beam_state.open_nt_count == 1: # check to make sure there are no remaining words
          assert(beam_state.term_count == len_terms)
        new_open_nt_stack = new_open_nt_stack.cdr
        new_open_nt_count -= 1
        new_cons_nt_count = 0
        assert(new_open_nt_count >= 0)
        assert((new_open_nt_stack is None) == (new_open_nt_count == 0))

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
        action_score=beam_state.score + action_score
      )
      return new_state


    initial_beam_state = BeamState(prev_beam_state=None,
                                   lstm_state=initial_lstm_state,
                                   prev_word_index=self.eos_index,
                                   open_nt_stack=None,
                                   open_nt_count=0,
                                   cons_nt_count=0,
                                   action_count=0,
                                   term_count=0,
                                   prev_action_type=None,
                                   prev_action_index=self.eos_index,
                                   score=0)

    beam = []
    completed = []

    while len(completed < beam_size and beam):
      successors = []

      while beam:
        current_beam_item = beam.pop()

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
