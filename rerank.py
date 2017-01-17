from __future__ import absolute_import, division, print_function
from utils import PTBModel, MediumConfig

import sys, time
import pickle
import numpy as np
import tensorflow as tf

import reader

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "medium",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None, "data_path")
flags.DEFINE_string('model_path', None, 'model_path')
flags.DEFINE_string('nbest_path', None, 'nbest_path')
flags.DEFINE_string('train_path', None, 'train_path')
flags.DEFINE_string('likelihood_file', None, 'likelihood_file')
flags.DEFINE_boolean('nbest', False, 'nbest')

FLAGS = flags.FLAGS


def score_all_trees(session, m, nbest, eval_op, eos, likelihood_file=None):
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

      if loss[i][j] < min_val:
        min_val = loss[i][j]
        ag = j
    if good:
      if FLAGS.nbest:
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


def rerank():
  config = pickle.load(open(FLAGS.model_path + '.config', 'rb'))
  config.batch_size = 10
  test_nbest_data, vocab = reader.ptb_raw_data2(FLAGS.data_path,
                                                FLAGS.nbest_path,
                                                train_path=FLAGS.train_path,
                                                remove_duplicates=FLAGS.likelihood_file is None)
  with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      m = PTBModel(is_training=False, config=config)

    saver = tf.train.Saver()
    saver.restore(session, FLAGS.model_path)
    score_all_trees(session, m, test_nbest_data, tf.no_op(), vocab['<eos>'], likelihood_file=FLAGS.likelihood_file)


def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")
  if not FLAGS.nbest_path:
    raise ValueError("Must set --nbest_path to nbest data")
  if not FLAGS.model_path:
    raise ValueError("Must set --model_path to model")
  rerank()


if __name__ == "__main__":
  tf.app.run()
