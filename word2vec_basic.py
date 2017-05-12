
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile
import numpy as np
from six.moves import urllib
from six.moves import xrange
import tensorflow as tf
import tensorflow_func as tf_func
import pickle

url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
  if not os.path.exists(filename):
    filename, _ = urllib.request.urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    print(statinfo.st_size)
    raise Exception(
        'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

filename = maybe_download('text8.zip', 31344016)


def read_data(filename):
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data

words = read_data(filename)
print('Data size', len(words))
vocabulary_size = 200000

def build_dataset(words):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
del words
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

unigram_cnt = [c for w, c in count]
total = sum(unigram_cnt)
unigram_prob = [c*1.0/total for c in unigram_cnt]

data_index = 0

def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  scope = 2 * skip_window + 1
  buff = collections.deque(maxlen=scope)
  for _ in range(scope):
      buff.append(data[data_index])
      data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
      target = skip_window
      targets_to_avoid = [skip_window]
      for j in range(num_skips):
          #target = 0;
          while target in targets_to_avoid:
              #target = target + 1;
              target = random.randint(0, scope - 1)
          targets_to_avoid.append(target)
          batch[i * num_skips + j] = buff[skip_window]
          labels[i * num_skips + j, 0] = buff[target]
      buff.append(data[data_index])
      data_index = (data_index + 1) % len(data)
  data_index = (data_index + len(data) - scope) % len(data)
  return batch, labels

batch_size = 100
embedding_size = 128
skip_window = 1
num_skips = 2
valid_size = 16
valid_window = 128
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64

graph = tf.Graph()

with graph.as_default():
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
  with tf.device('/cpu:0'):
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    biases = tf.Variable(tf.zeros([vocabulary_size]))

    true_w = tf.nn.embedding_lookup(weights, train_labels)
    true_w = tf.reshape(true_w, [-1, embedding_size])

    sample = np.random.choice(vocabulary_size, num_sampled, p=unigram_prob, replace=False)

  loss = tf.reduce_mean(tf_func.nce_loss(embed, weights, biases, train_labels, sample))
  #loss = tf.reduce_mean(tf_func.cross_entropy_loss(embed, true_w))

  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)
  init = tf.global_variables_initializer()

num_steps = 100001

with tf.Session(graph=graph) as session:
  init.run()
  print("Initialized")
  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      print("Average loss at step ", step, ": ", average_loss)
      average_loss = 0
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        if (valid_word == 'first' or valid_word == 'american' or valid_word == 'would'):
            top_k = 20
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = "Result for %s:" % valid_word
            for k in xrange(top_k):
                close_word = reverse_dictionary[nearest[k]]
                log_str = "%s %s," % (log_str, close_word)
            print(log_str)
  final_embeddings = normalized_embeddings.eval()
  pickle.dump([dictionary, final_embeddings], open('Models_nce/word2vec2.model', 'wb'))

