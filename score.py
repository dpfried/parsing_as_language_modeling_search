import numpy as np
import reader
import sys
import tensorflow as tf
import utils

def score_single_tree(session, m, x):
  # assert(m.batch_size == 1)
  # assert(m.num_steps == 1)
  state = []
  total_cost = 0.0
  for c, h in m.initial_state:
    state.append((c.eval(), h.eval()))

  costs = []

  for i in range(len(x) - 1):
    input_arr = x[i:i+1][:,None]
    target_arr = x[i+1:i+2][:,None]
    # input_arr = x[i]
    # target_arr = x[i+1]
    fetches = [m.cost]
    for c, h in m.final_state:
      fetches.append(c)
      fetches.append(h)
    feed_dict = {
      m.input_data: input_arr,
      m.targets: target_arr,
      m.weights: np.ones((m.batch_size, m.num_steps), dtype=np.float32)
    }
    for (state_c, state_h), (c, h) in zip(state, m.initial_state):
      feed_dict[c] = state_c
      feed_dict[h] = state_h
    res = session.run(fetches, feed_dict)
    cost = res[0]
    costs.append(cost[0])
    state_flat = res[1:] # [c1, h1, c2, h2...]
    state = [state_flat[i:i+2] for i in range(0, len(state_flat), 2)]
    total_cost += np.sum(cost)
  # print(costs)
  return total_cost

def score_trees_separate_batching(session, model, trees_list, eval_op, eos_index):
    all_losses = []
    for xyms in utils.separate_trees_iterator(trees_list, eos_index, model.batch_size, model.num_steps):
        losses = np.zeros(model.batch_size)

        state = []
        for c, h in model.initial_state: # initial_state: ((c1, m1), (c2, m2))
            state.append((c.eval(), h.eval()))

        # x: inputs
        # y: targets:
        # m: mask (1 if the corresponding x&y are part of a sentence, 0 otherwise)
        for (x, y, m) in xyms:
            fetches = []
            fetches.append(model.cost)
            fetches.append(eval_op)
            for c, h in model.final_state: # final_state: ((c1, m1), (c2, m2))
                fetches.append(c)
                fetches.append(h)
            feed_dict = {}
            feed_dict[model.input_data] = x
            feed_dict[model.targets] = y
            feed_dict[model.weights] = m
            for k, (c, h) in enumerate(model.initial_state):
                feed_dict[c], feed_dict[h] = state[k]
            res = session.run(fetches, feed_dict)
            cost = res[0]
            state_flat = res[2:] # [c1, m1, c2, m2]
            state = [state_flat[i:i+2] for i in range(0, len(state_flat), 2)]

            cost = cost.reshape((model.batch_size, model.num_steps))
            losses += np.sum(cost * m, 1)

        all_losses.extend(losses)

    return all_losses[:len(trees_list)]


def score_all_trees(session, m, nbest, eval_op, eos, likelihood_file=None, output_nbest=False):
  """Runs the model on the given data."""
  counts = []
  loss = []
  prev = (-1, -1)
  for pair in nbest['idx2tree']:
    if pair[0] != prev[0]:
      counts.append([0])
      loss.append([0.])
    elif pair[1] == prev[1] + 1:
      counts[-1].append(0)
      loss[-1].append(0.)
    counts[-1][-1] += 1
    prev = pair
  data = nbest['data']
  epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
  costs = 0.0
  iters = 0
  state = []

  weights = np.ones((m.batch_size, m.num_steps), dtype=np.float32)
  for c, h in m.initial_state: # initial_state: ((c1, m1), (c2, m2))
    state.append((c.eval(), h.eval()))
  for step, (x, y, z) in enumerate(
      reader.ptb_iterator2(data, m.batch_size, m.num_steps,
                           nbest['idx2tree'], eos)):
    sys.stderr.write("\r%s" % step)
    fetches = []
    fetches.append(m.cost)
    fetches.append(eval_op)
    for c, h in m.final_state: # final_state: ((c1, m1), (c2, m2))
      fetches.append(c)
      fetches.append(h)
    feed_dict = {}
    feed_dict[m.input_data] = x
    feed_dict[m.targets] = y
    feed_dict[m.weights] = weights
    for i, (c, h) in enumerate(m.initial_state):
      feed_dict[c], feed_dict[h] = state[i]
    res = session.run(fetches, feed_dict)
    cost = res[0]
    state_flat = res[2:] # [c1, m1, c2, m2]
    state = [state_flat[i:i+2] for i in range(0, len(state_flat), 2)]
    costs += np.sum(cost) / m.batch_size
    iters += m.num_steps

    cost = cost.reshape((m.batch_size, m.num_steps))
    for idx, val in np.ndenumerate(cost):
      tree_idx = z[idx[0]][idx[1]]
      if tree_idx[0] == -1: # dummy
        continue
      counts[tree_idx[0]][tree_idx[1]] -= 1
      loss[tree_idx[0]][tree_idx[1]] += cost[idx[0]][idx[1]]

  trees = nbest['trees']
  bad = []
  num_words = 0
  if likelihood_file is not None:
    f_lik = open(likelihood_file, 'w')
  else:
    f_lik = None
  for i in range(len(trees)):
    good = True
    ag = 0
    min_val = float('inf')
    if f_lik is not None:
      f_lik.write("%s\n" % len(trees[i]))
    for j in range(len(trees[i])):
      if counts[i][j] != 0:
        bad.append(i)
        good = False
        break

      if f_lik is not None:
        f_lik.write("%s\n" % -loss[i][j])

      sys.stderr.write("(%d, %d): %s\n" % (i, j, loss[i][j]))
      if loss[i][j] < min_val:
        min_val = loss[i][j]
        ag = j
    if good:
      if output_nbest:
        print(len(trees[i]))
        for j in range(len(trees[i])):
          print(loss[i][j])
          print(trees[i][j])
        print()
      else:
        print(trees[i][ag])

  if f_lik is not None:
    f_lik.close()
  if bad:
    print('bad: %s' % ', '.join([str(x) for x in bad]))
    return None
  else:
    return loss
