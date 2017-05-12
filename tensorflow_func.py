import tensorflow as tf
import collections
import numpy
def cross_entropy_loss(inputs, true_w):
    tmp = tf.matmul(true_w,tf.transpose(inputs))
    A = tf.diag_part(tmp)
    B = tf.log(tf.reduce_sum(tf.exp(tmp),1))
    return tf.subtract(B, A)

def nce_loss(inputs, weights, biases, labels, sample):
    embedding_size = 128
    true_w = tf.nn.embedding_lookup(weights, labels)
    true_w = tf.reshape(true_w, [-1, embedding_size])
    max = tf.matmul(inputs, tf.transpose(true_w))
    tmp_a = tf.sigmoid(tf.diag_part(max))
    A = tf.log(tmp_a)
    sample_w = tf.nn.embedding_lookup(weights, sample)
    tmp_b = tf.sigmoid(tf.matmul(inputs, tf.transpose(sample_w)))
    one = tf.ones([100, 64], tf.float32)
    B = tf.reduce_sum(tf.log(tf.subtract(one, tmp_b)), 1)
    return -tf.add(A, B)

