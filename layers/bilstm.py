# -*- coding: utf-8 -*-
"""
 Time : 2021/6/27 11:44
 Author : huangkai
 File : bilstm.py
 Software: PyCharm 
 mail:18707125049@163.com
 
"""

import tensorflow as tf


class BiLSTM(object):
    """
    A LSTM based deep Siamese network for text similarity.
    Uses an character embedding layer, followed by a biLSTM and Energy Loss layer.
    """

    # def __init__(
    #         self, embedding1, embedding2, hidden_units, dropout_keep_prob):

    # Placeholders for input, output and dropout

    # Create a convolution + maxpool layer for each filter size
    # with tf.name_scope("output"):
    #     out1 = self.model(embedding1, dropout_keep_prob, "side1", hidden_units)
    #     out2 = self.model(embedding2, dropout_keep_prob, "side2", hidden_units)
    #     print(out1.shape, out2.shape)
    # self.output = tf.concat([tf.multiply(out1, out2), tf.abs(tf.subtract(out1, out2))], axis=-1)
    # self.output = self.cosine(out1, out2)
    # print(self.output.shape)

    @staticmethod
    def model(x, dropout, hidden_units):
        n_layers = 3
        stacked_rnn_fw = []
        stacked_rnn_bw = []
        for _i in range(n_layers):
            fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_units)
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=dropout)
            stacked_rnn_fw.append(fw_cell)
        lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)

        for _i in range(n_layers):
            bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_units)
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=dropout)
            stacked_rnn_bw.append(bw_cell)
        lstm_bw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_bw, state_is_tuple=True)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x, dtype=tf.float32)
        return tf.concat(outputs, 2)

    def cosine(self, q, a):
        pooled_len_1 = tf.sqrt(tf.reduce_sum(q * q, 1))
        pooled_len_2 = tf.sqrt(tf.reduce_sum(a * a, 1))
        pooled_mul_12 = tf.reduce_sum(q * a, 1)
        score = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 + 1e-8, name="scores")
        return score
