from __future__ import absolute_import, division, print_function
from utils import PTBModel, MediumConfig

import sys, time
import pickle
import numpy as np
import tensorflow as tf

from score import score_all_trees

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
    score_all_trees(session, m, test_nbest_data, tf.no_op(), vocab['<eos>'], likelihood_file=FLAGS.likelihood_file, output_nbest=FLAGS.nbest)


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
