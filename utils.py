import collections, gzip, time
import numpy as np
import tensorflow as tf
import utils
import sys


OPTIMIZERS = ['sgd', 'adam', 'sgd_momentum']

class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 0.25
  max_grad_norm = 20
  num_layers = 3
  num_steps = 50
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 50
  keep_prob = 0.3
  # correction: for wsj model, we use 0.9.
  lr_decay = 0.9
  batch_size = 20
  downscale_loss_by_num_steps = False
  optimizer = 'sgd'


class PTBModel(object):
  def __init__(self, is_training, config):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

    self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])
    self._weights = tf.placeholder(tf.float32, [batch_size, num_steps])

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=1.0,
                                             state_is_tuple=True)
    if is_training and config.keep_prob < 1:
      lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
          lstm_cell, output_keep_prob=config.keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers,
                                       state_is_tuple=True)

    self._initial_state = cell.zero_state(batch_size, tf.float32)

    with tf.device("/cpu:0"):
      embedding = tf.get_variable("embedding", [vocab_size, size])
      inputs = tf.nn.embedding_lookup(embedding, self._input_data)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    inputs = [tf.squeeze(input_, [1])
              for input_ in tf.split(1, num_steps, inputs)]
    self._inputs = inputs
    outputs, state = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)
    self._outputs = outputs
    self._state = state

    output = tf.reshape(tf.concat(1, outputs), [-1, size])
    self._output = output
    softmax_w = tf.get_variable("softmax_w", [size, vocab_size])
    softmax_b = tf.get_variable("softmax_b", [vocab_size])
    logits = tf.matmul(output, softmax_w) + softmax_b
    self._logits = logits
    loss = tf.nn.seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(self._targets, [-1])],
        [tf.reshape(self._weights, [-1])])
    self._loss = loss
    self._log_probs = tf.nn.log_softmax(logits)
    if config.downscale_loss_by_num_steps:
      print("batch loss will be normalized by number of tokens")
      cost = tf.reduce_sum(loss) / tf.reduce_sum(self._weights)
    else:
      print("batch loss will be normalized by batch size")
      cost = tf.reduce_sum(loss) / batch_size
    self._cost = loss
    self._final_state = state

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    if config.optimizer == 'sgd':
      optimizer = tf.train.GradientDescentOptimizer(self.lr)
    elif config.optimizer == 'adam':
      optimizer = tf.train.AdamOptimizer(self.lr)
    elif config.optimizer == 'sgd_momentum':
      optimizer = tf.train.MomentumOptimizer(self.lr, 0.9)
    else:
      raise ValueError("invalid optimizer %s" % config.optimizer)

    self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    self.downscale_loss_by_num_steps = config.downscale_loss_by_num_steps

  def assign_lr(self, session, lr_value):
    session.run(tf.assign(self.lr, lr_value))

  @property
  def input_data(self):
    return self._input_data

  @property
  def targets(self):
    return self._targets

  @property
  def weights(self):
    return self._weights

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def log_probs(self):
    return self._log_probs

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op


def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id


def _read_words(filename):
  with open_file(filename) as f:
    return f.read().replace('\n', '<eos>').split()


def chop(data, eos, prepend_eos=False):
  new_data = []
  sent = []
  if prepend_eos:
    sent.append(eos)
  for w in data:
    sent.append(w)
    if w == eos:
      new_data.append(sent)
      sent = []
      if prepend_eos:
        sent.append(eos)
  return new_data


def open_file(path):
  if path.endswith('.gz'):
    return gzip.open(path, 'rb')
  else:
    return open(path, 'r')


# iterator used for nbest data.
def ptb_iterator2(raw_data, batch_size, num_steps, idx2tree, eos):
  dummy1 = 0
  dummy2 = (-1, -1)
  remainder = len(raw_data) % batch_size
  if remainder != 0:
    raw_data = raw_data + [dummy1 for x in range(batch_size - remainder)]
    idx2tree = idx2tree + [dummy2 for x in range(batch_size - remainder)]
  raw_data = np.array(raw_data, dtype=np.int32)

  data_len = len(raw_data)
  batch_len = data_len // batch_size
  remainder = (data_len // batch_size) % num_steps

  data = np.zeros([batch_size, batch_len + num_steps - remainder + 1],
                  dtype=np.int32)
  for i in range(batch_size):
    data[i, 1:batch_len+1] = raw_data[batch_len * i:batch_len * (i + 1)]
    if i == 0:
      data[i, 0] = eos
    else:
      # TODO: should be batch_len*i - 1
      data[i, 0] = raw_data[batch_len - 1]
  idx2tree = np.array(idx2tree, dtype=np.dtype('int, int'))
  tree = np.zeros([batch_size, batch_len + num_steps - remainder],
                  dtype=np.dtype('int, int'))
  for i in range(batch_size):
    tree[i, :batch_len] = idx2tree[batch_len * i:batch_len * (i + 1)]
    tree[i, batch_len:] = [dummy2 for x in range(num_steps - remainder)]

  epoch_size = (batch_len + num_steps - remainder) // num_steps

  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  for i in range(epoch_size):
    x = data[:, i*num_steps:(i+1)*num_steps]
    y = data[:, i*num_steps+1:(i+1)*num_steps+1]
    z = tree[:, i*num_steps:(i+1)*num_steps]
    yield (x, y, z)


def run_epoch(session, m, data, eval_op, verbose=False):
  """Runs the model on the given data."""
  epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = []
  for c, h in m.initial_state: # initial_state: ((c1, m1), (c2, m2))
    state.append((c.eval(), h.eval()))
  weights = np.ones((m.batch_size, m.num_steps), dtype=np.float32)
  for step, (x, y) in enumerate(ptb_iterator(data, m.batch_size,
                                             m.num_steps)):
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

    if verbose and step % (epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / epoch_size, np.exp(costs / iters),
             iters * m.batch_size / (time.time() - start_time)))

  return np.exp(costs / iters)

def run_epoch_separate_batched(session, model, data, eval_op, eos_index, verbose=False):
  """Runs the model on the given data."""
  costs = 0.0
  iters = 0
  trees_list = chop(data, eos_index, prepend_eos=True)
  epoch_size = len(trees_list) // model.batch_size

  start_time = time.time()
  for step, xyms in enumerate(utils.separate_trees_iterator(trees_list, eos_index, model.batch_size, model.num_steps)):
    state = []
    for c, h in model.initial_state: # initial_state: ((c1, m1), (c2, m2))
      state.append((c.eval(), h.eval()))
    for x, y, m in xyms:
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
      for i, (c, h) in enumerate(model.initial_state):
        feed_dict[c], feed_dict[h] = state[i]
      res = session.run(fetches, feed_dict)
      cost = res[0]
      state_flat = res[2:] # [c1, m1, c2, m2]
      state = [state_flat[i:i+2] for i in range(0, len(state_flat), 2)]
      # for a, b, c in zip(x, m, cost.reshape(model.batch_size, model.num_steps)):
      #     print("x", a)
      #     print("m", b)
      #     print("c", c)
      #     print
      # print
      costs += np.sum(cost)
      iters += np.sum(m)

    num_tokens = sum(len(l) - 1 for l in trees_list[:(step+1) * model.batch_size])
    assert(num_tokens == iters)

    if verbose and step % (epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / epoch_size, np.exp(costs / iters),
             iters / (time.time() - start_time)))

  # print("total steps", iters)
  return np.exp(costs / iters)


def run_epoch2(session, m, nbest, eval_op, eos, verbose=False):
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
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = []
  weights = np.ones((m.batch_size, m.num_steps), dtype=np.float32)
  for c, h in m.initial_state: # initial_state: ((c1, m1), (c2, m2))
    state.append((c.eval(), h.eval()))
  for step, (x, y, z) in enumerate(
          ptb_iterator2(data, m.batch_size, m.num_steps,
                        nbest['idx2tree'], eos)):
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

    if verbose and step % (epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / epoch_size, np.exp(costs / iters),
             iters * m.batch_size / (time.time() - start_time)))

  scores = nbest['scores']
  num = 0
  gold, test, matched = 0, 0, 0
  bad = []
  for i in range(len(scores)):
    good = True
    ag = 0
    min_val = 10000000
    for j in range(len(scores[i])):
      if counts[i][j] != 0:
        bad.append(i)
        good = False
        break
      if loss[i][j] < min_val:
        min_val = loss[i][j]
        ag = j
    if good:
      num += 1
      gold += scores[i][ag]['gold']
      test += scores[i][ag]['test']
      matched += scores[i][ag]['matched']
  if bad:
    print('bad: %s' % ', '.join([str(x) for x in bad]))
  return 200. * matched / (gold + test), num

def run_epoch2_separate_batched(session, m, nbest, eval_op, eos, verbose=False):
  import score
  """Runs the model on the given data."""
  data = nbest['data']
  scores = nbest['scores']
  split_nbest = chop(data, eos, prepend_eos=True)
  assert(len(split_nbest) == sum(len(s) for s in scores))
  losses = score.score_trees_separate_batching(session, m, split_nbest, eval_op, eos)
  assert(len(split_nbest) == len(losses))

  unflattened_losses = []
  counter = 0
  for sc in scores:
    next_counter = counter + len(sc)
    unflattened_losses.append(losses[counter:next_counter])
    counter = next_counter
  assert(len(unflattened_losses) == len(scores))

  num = len(unflattened_losses)
  gold, test, matched = 0, 0, 0
  for l, sc in zip(unflattened_losses, scores):
    best_loss, best_score = min(zip(l, sc), key=lambda p: p[0])
    gold += best_score['gold']
    test += best_score['test']
    matched += best_score['matched']
  return 200. * matched / (gold + test), num


def unkify(ws):
  uk = 'unk'
  sz = len(ws)-1
  if ws[0].isupper():
    uk = 'c' + uk
  if ws[0].isdigit() and ws[sz].isdigit():
    uk = uk + 'n'
  elif sz <= 2:
    pass
  elif ws[sz-2:sz+1] == 'ing':
    uk = uk + 'ing'
  elif ws[sz-1:sz+1] == 'ed':
    uk = uk + 'ed'
  elif ws[sz-1:sz+1] == 'ly':
    uk = uk + 'ly'
  elif ws[sz] == 's':
    uk = uk + 's'
  elif ws[sz-2:sz+1] == 'est':
    uk = uk + 'est'
  elif ws[sz-1:sz+1] == 'er':
    uk = uk + 'ER'
  elif ws[sz-2:sz+1] == 'ion':
    uk = uk + 'ion'
  elif ws[sz-2:sz+1] == 'ory':
    uk = uk + 'ory'
  elif ws[0:2] == 'un':
    uk = 'un' + uk
  elif ws[sz-1:sz+1] == 'al':
    uk = uk + 'al'
  else:
    for i in range(sz):
      if ws[i] == '-':
        uk = uk + '-'
        break
      elif ws[i] == '.':
        uk = uk + '.'
        break
  return '<' + uk + '>'

def convert_to_ptb_format(id_to_token, indices, gold_tokens=None, gold_tags=None):
  indices = list(indices)
  if id_to_token[indices[0]] == '<eos>':
    indices.pop(0)
  if id_to_token[indices[-1]] == '<eos>':
    indices.pop()

  ptb_tokens = []
  stack = []
  word_ix = 0
  for id_ in indices:
    token = id_to_token[id_]
    if token.startswith('('):
      nt = token[1:]
      stack.append(nt)
      ptb_tokens.append(token)
    elif token.startswith(')'):
      nt = token[1:]
      assert(nt == stack[-1])
      stack.pop()
      ptb_tokens.append(')')
    else:
      # create pos tags above the terminal
      if gold_tokens is not None:
        token_to_print = gold_tokens[word_ix]
      else:
        token_to_print = token
      if gold_tags is not None:
        tag_to_print = gold_tags[word_ix]
      else:
        tag_to_print = "XX"
      ptb_tokens.extend(["(%s %s)" % (tag_to_print, token_to_print)])
      word_ix += 1
  assert(not stack)
  if gold_tokens:
    assert(word_ix == len(gold_tokens))
  if gold_tags:
    assert(word_ix == len(gold_tags))
  return ptb_tokens

def ptb_iterator(raw_data, batch_size, num_steps):
  raw_data = np.array(raw_data, dtype=np.int32)

  data_len = len(raw_data)
  batch_len = data_len // batch_size
  data = np.zeros([batch_size, batch_len], dtype=np.int32)
  for i in range(batch_size):
    data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

  epoch_size = (batch_len - 1) // num_steps

  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  for i in range(epoch_size):
    x = data[:, i*num_steps:(i+1)*num_steps]
    y = data[:, i*num_steps+1:(i+1)*num_steps+1]
    yield (x, y)

def separate_trees_iterator(separate_trees, eos_index, batch_size, num_steps):
  # given a list of lists of token indices (one list per parse), return an iterator over lists
  #  (x, y, m), where x is inputs, y is targets, m is a mask, and all are dim (batch_size, num_steps)
  # we return lists of (x, y, m) in case some sentence within that batch is longer than num_steps
  # in that case, the hidden states should be passed between the lstm application to each tuple in the list
  for tree in separate_trees:
    assert(tree[0] == eos_index)
    assert(tree[-1] == eos_index)
  for sent_offset in range(0, len(separate_trees), batch_size):
    batch = separate_trees[sent_offset:sent_offset+batch_size]
    if len(batch) < batch_size:
      batch += [[]] * (batch_size - len(batch))

    assert(len(batch) == batch_size)

    # get the smallest multiple of num_steps which is at least the length of the longest sentence minus one (since we will zip source and targets)
    width = ((max(len(x) - 1 for x in batch) + num_steps - 1)  // num_steps) * num_steps
    # pad sequences
    mask = np.zeros((batch_size, width), dtype=np.float32)
    padded = np.zeros((batch_size, width + 1), dtype=np.int32)

    for row, tree in enumerate(batch):
      mask[row,:len(tree)-1] = 1
      padded[row,:len(tree)] = tree

    all_xym = []
    for j in range(0, width, num_steps):
      x = padded[:,j:j+num_steps]
      y = padded[:,j+1:j+num_steps+1]
      m = mask[:,j:j+num_steps]
      assert(x.shape == y.shape)
      assert(m.shape == y.shape)
      assert(m.shape == (batch_size, num_steps))
      all_xym.append((x,y,m))

    yield all_xym
