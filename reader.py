from __future__ import absolute_import, division, print_function
from utils import _build_vocab, _read_words, open_file, unkify

import os
import numpy as np
import sys


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data]


# read preprocessed nbest
def _file_to_word_ids2(filename, word_to_id):
  data = []
  scores = []
  nbest = []
  idx2tree = []
  count = 0
  with open_file(filename) as f:
    for line in f:
      if count == 0:
        count = int(line)
      elif not line.startswith(' '):
        tmp = line.split()
        gold = int(tmp[0])
        test = int(tmp[1])
        matched = int(tmp[2])
      else:
        line = line.replace('\n', '<eos>').split()
        line = [word_to_id[word] for word in line]
        for i in range(len(line)):
          idx2tree.append((len(scores), len(nbest)))
        nbest.append({'gold': gold, 'test': test, 'matched': matched})
        count -= 1
        data.extend(line)
        if count == 0:
          scores.append(nbest)
          nbest = []
  return {'data': data, 'scores': scores, 'idx2tree': idx2tree}


def _file_to_word_ids3(filename, word2id, remove_duplicates=True, sent_limit=None):
  data = []
  trees = []
  idx2tree = []
  for sent_count, ts in enumerate(_generate_nbest(open_file(filename))):
    if sent_limit and sent_count >= sent_limit:
      break
    for t in ts:
      t['seq'] = _process_tree(t['ptb'], word2id)
    if remove_duplicates:
      ts = _remove_duplicates(ts)
    nbest = []
    for t in ts:
      nums = [word2id[word] for word in t['seq'].split() + ['<eos>']]
      for i in range(len(nums)):
        idx2tree.append((len(trees), len(nbest)))
      nbest.append(t['ptb'])
      data.extend(nums)
    trees.append(nbest)
  return {'data': data, 'trees': trees, 'idx2tree': idx2tree}

def ptb_list_to_word_ids(parses_by_sent, word2id, remove_duplicates=True, sent_limit=None):
  data = []
  trees = []
  idx2tree = []
  dict_format = [[{'ptb': parse} for parse in parses] for parses in parses_by_sent]
  for sent_count, ts in enumerate(dict_format):
    if sent_limit and sent_count >= sent_limit:
      break
    for t in ts:
      t['seq'] = _process_tree(t['ptb'], word2id)
    if remove_duplicates:
      ts = _remove_duplicates(ts)
    nbest = []
    for t in ts:
      nums = [word2id[word] for word in t['seq'].split() + ['<eos>']]
      for i in range(len(nums)):
        idx2tree.append((len(trees), len(nbest)))
      nbest.append(t['ptb'])
      data.extend(nums)
    trees.append(nbest)
  return {'data': data, 'trees': trees, 'idx2tree': idx2tree}


def _generate_nbest(f):
  nbest = []
  count = 0
  for line in f:
    # line=line.strip()
    line = line[:-1]
    if line == '':
      continue
    if count == 0:
      count = int(line.split()[0])
    elif line.startswith('('):
      nbest.append({'ptb': line})
      count -= 1
      if count == 0:
        yield nbest
        nbest = []


def _process_tree(line, words, tags=False):
  tokens = line.replace(')', ' )').split()
  nonterminals = []
  new_tokens = []
  pop = False
  ind = 0
  for token in tokens:
    if token.startswith('('): # open paren
      new_token = token[1:]
      nonterminals.append(new_token)
      new_tokens.append(token)
    elif token == ')': # close paren
      if pop: # preterminal
        pop = False
      else: # nonterminal
        new_token = ')' + nonterminals.pop()
        new_tokens.append(new_token)
    else: # word
      if not tags:
        tag = '(' + nonterminals.pop() # pop preterminal
        new_tokens.pop()
        pop = True
      if token.lower() in words:
        new_tokens.append(token.lower())
      else:
        new_tokens.append(unkify(token))
  return ' ' + ' '.join(new_tokens[1:-1]) + ' '


def _remove_duplicates(nbest):
  new_nbest = []
  seqs = set()
  for t in nbest:
    if t['seq'] not in seqs:
      seqs.add(t['seq'])
      new_nbest.append(t)
  return new_nbest


# read silver data
def file_to_word_ids3(filename):
  for line in open_file(filename):
    yield [int(x) for x in line.split()]


# read data for training.
def ptb_raw_data(data_path=None, train_path=None, valid_path=None, valid_nbest_path=None):
  if train_path is None:
    train_path = os.path.join(data_path, "train.gz")
  if valid_path is None:
    valid_path = os.path.join(data_path, "dev.gz")
  if valid_nbest_path is None:
    valid_nbest_path = os.path.join(data_path, "dev_nbest.gz")

  word_to_id = _build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  valid_nbest_data = _file_to_word_ids2(valid_nbest_path, word_to_id)
  return train_data, valid_data, valid_nbest_data, word_to_id


# read data for reranking.
def ptb_raw_data2(data_path=None, nbest_path=None, train_path=None, remove_duplicates=True, sent_limit=None):
  if train_path is None:
    train_path = os.path.join(data_path, "train.gz")
  word_to_id = _build_vocab(train_path)
  nbest_data = _file_to_word_ids3(nbest_path, word_to_id, remove_duplicates=remove_duplicates, sent_limit=sent_limit)
  return nbest_data, word_to_id

# read data for tri-training.
def ptb_raw_data3(data_path=None, train_path=None, valid_path=None, valid_nbest_path=None, silver_path=None):
  if train_path is None:
    train_path = os.path.join(data_path, "train.gz")
  if silver_path is None:
    silver_path = os.path.join(data_path, 'silver.gz')
  if valid_path is None:
    valid_path = os.path.join(data_path, "dev.gz")
  if valid_nbest_path is None:
    valid_nbest_path = os.path.join(data_path, "dev_nbest.gz")

  word_to_id = _build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  valid_nbest_data = _file_to_word_ids2(valid_nbest_path, word_to_id)
  return train_data, silver_path, valid_data, valid_nbest_data, word_to_id


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
      # todo: should be raw_data[batch_len*i - 1] ?
      data[i, 0] = raw_data[batch_len - 1]
  idx2tree = np.array(idx2tree, dtype=np.dtype('int, int'))
  tree = np.zeros([batch_size, batch_len + num_steps - remainder],
                  dtype=np.dtype('int, int'))
  for i in range(batch_size):
    tree[i, :batch_len] = idx2tree[batch_len * i:batch_len * (i + 1)]
    tree[i, batch_len:] = [dummy2 for x in range(num_steps - remainder)]

  epoch_size = (batch_len + num_steps - remainder) // num_steps

  sys.stderr.write("epoch size: %s\n" % epoch_size)

  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  for i in range(epoch_size):
    x = data[:, i*num_steps:(i+1)*num_steps]
    y = data[:, i*num_steps+1:(i+1)*num_steps+1]
    z = tree[:, i*num_steps:(i+1)*num_steps]
    yield (x, y, z)

def ptb_iterator2_single_sentence(raw_data, idx2tree, eos):
  last_idx = None
  this_x = [eos]

  assert(len(raw_data) == len(idx2tree))

  for id_, idx in zip(raw_data, idx2tree):
    if last_idx is not None and idx != last_idx:
      yield np.array(this_x)
      this_x = [eos]

    this_x.append(id_)
    last_idx = idx

  if this_x:
    yield np.array(this_x)

def ptb_iterator_single_sentence(raw_data, eos):
  raw_data = np.array(raw_data, dtype=np.int32)

  this_x = [eos]

  for id_ in raw_data:
    this_x.append(id_)
    if id_ == eos:
      yield np.array(this_x)
      this_x = [eos]

  if len(this_x) > 1:
    yield np.array(this_x)
