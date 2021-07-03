# -*- coding: utf-8 -*-
"""
 Time : 2021/6/27 12:06
 Author : huangkai
 File : siamese.py
 mail:18707125049@163.com
 paper:https://www.aclweb.org/anthology/W16-1617.pdf
"""
import tensorflow as tf
from args import siamese_args
from layers.bilstm import BiLSTM


class Graph:
    def __init__(self, embedding_type="ONE_HOT", embedding=None):
        self.q = tf.placeholder(dtype=tf.int32, shape=(None, siamese_args.seq_length), name='q')
        self.d = tf.placeholder(dtype=tf.int32, shape=(None, siamese_args.seq_length), name='d')
        self.y = tf.placeholder(dtype=tf.int32, shape=None, name='y')

        self.keep_prob = tf.placeholder(dtype=tf.float32, name='drop_rate')
        #
        if embedding_type == "ONE_HOT":
            self.embedding = tf.get_variable(dtype=tf.float32,
                                             shape=(siamese_args.vocab_size, siamese_args.char_embedding_size),
                                             name='embedding',
                                             trainable=True)
        elif embedding_type == "WORD2VEC":
            self.embedding = tf.get_variable(dtype=tf.float32, initializer=embedding, trainable=False)
        self.forward()

    def dropout(self, x):
        return tf.nn.dropout(x, keep_prob=self.keep_prob)

    # def bilstm(self, x, hidden_size):
    #     fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size,reuse=True)
    #     bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
    #
    #     return tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, dtype=tf.float32)
    @staticmethod
    def contrastive_loss(y, d):
        tmp = (1 - y) * tf.square(d)
        # tmp= tf.mul(y,tf.square(d))
        tmp2 = y * tf.square(tf.maximum((1 - d), 0)) / 4
        return tmp + tmp2

    @staticmethod
    def cosine(p, h):
        p_norm = tf.norm(p, axis=1, keepdims=True)
        h_norm = tf.norm(h, axis=1, keepdims=True)

        cosine = tf.reduce_sum(tf.multiply(p, h), axis=1, keepdims=True) / (p_norm * h_norm)

        return cosine

    def forward(self):
        p_embedding = tf.nn.embedding_lookup(self.embedding, self.q)
        h_embedding = tf.nn.embedding_lookup(self.embedding, self.d)

        with tf.variable_scope("lstm_p"):
            out1 = BiLSTM.model(x=p_embedding,
                                dropout=self.keep_prob,
                                hidden_units=siamese_args.embedding_hidden_size)
        with tf.variable_scope("lstm_p", reuse=True):
            out2 = BiLSTM.model(x=h_embedding,
                                dropout=self.keep_prob,
                                hidden_units=siamese_args.embedding_hidden_size)
        output1 = tf.reduce_mean(out1, axis=1)  # temporal average
        output2 = tf.reduce_mean(out2, axis=1)
        print('output1', output1.shape)
        print('output2', output2.shape)
        # fc
        with tf.variable_scope("feedforward_128"):
            intermediate_output1 = tf.layers.dense(output1, 128, activation=tf.nn.relu)
            intermediate_output1 = tf.layers.dropout(intermediate_output1, rate=0.4)
            intermediate_output1 = tf.layers.batch_normalization(intermediate_output1)
        with tf.variable_scope("feedforward_128", reuse=True):
            intermediate_output2 = tf.layers.dense(output2, 128, activation=tf.nn.relu)
            intermediate_output2 = tf.layers.dropout(intermediate_output2, rate=0.4)
            intermediate_output2 = tf.layers.batch_normalization(intermediate_output2)
        output1 = tf.reshape(intermediate_output1, (-1, 128))
        output2 = tf.reshape(intermediate_output2, (-1, 128))
        print('output1', output1.shape)
        print('output2', output2.shape)
        result = self.cosine(output1, output2)
        neg_result = 1 - result
        print('result', result.shape)
        logits = tf.concat([result, neg_result], axis=1)

        self.train(logits)

    def train(self, logits):
        y = tf.one_hot(self.y, siamese_args.class_size)
        loss = self.contrastive_loss(y, logits)
        self.loss = tf.reduce_mean(loss)
        self.prediction = tf.argmax(logits, axis=1)
        self.train_op = tf.train.AdamOptimizer(siamese_args.learning_rate).minimize(self.loss)
        correct_prediction = tf.equal(tf.cast(self.prediction, tf.int32), self.y)
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
