# -*- coding: utf-8 -*-
"""
 Time : 2021/7/4 21:16
 Author : huangkai
 File : bimpm_v2.py
 mail:18707125049@163.com
 paper:https://arxiv.org/pdf/1702.03814.pdf
"""
import tensorflow as tf
from args import bimpm_args


class Graph:
    def __init__(self, embedding_type="ONE_HOT", embedding=None):
        self.p = tf.placeholder(dtype=tf.int32, shape=(None, bimpm_args.seq_length), name='p')
        self.h = tf.placeholder(dtype=tf.int32, shape=(None, bimpm_args.seq_length), name='h')
        self.y = tf.placeholder(dtype=tf.int32, shape=None, name='y')

        self.keep_prob = tf.placeholder(dtype=tf.float32, name='drop_rate')
        #
        if embedding_type == "ONE_HOT":
            self.embedding = tf.get_variable(dtype=tf.float32,
                                             shape=(bimpm_args.vocab_size, bimpm_args.char_embedding_size),
                                             name='embedding')
        elif embedding_type == "PRETRAIN_MODEL":
            self.embedding = tf.get_variable(dtype=tf.float32, initializer=embedding, trainable=False)
        self.forward()

    # def dropout(self, x):
    #     return tf.nn.dropout(x, keep_prob=self.keep_prob)

    @staticmethod
    def cosine(p, h):
        p_norm = tf.norm(p, axis=1, keepdims=True)
        h_norm = tf.norm(h, axis=1, keepdims=True)

        cosine = tf.reduce_sum(tf.multiply(p, h), axis=1, keepdims=True) / (p_norm * h_norm)

        return cosine

    def lstm(self, x):
        cell = tf.nn.rnn_cell.BasicLSTMCell(bimpm_args.char_hidden_size)
        return tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

    def bilstm(self, x):
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(bimpm_args.word_embedding_len)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(bimpm_args.word_embedding_len)
        return tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, dtype=tf.float32)

    def forward(self):
        p_embedding = tf.nn.embedding_lookup(self.embedding, self.p)
        h_embedding = tf.nn.embedding_lookup(self.embedding, self.h)

        p_context = self.fully_connect(p_embedding)
        h_context = self.fully_connect(h_embedding)

        # [0,1],[1,0]  [0,0,1]...
        pos_result = self.cosine(p_context, h_context)
        neg_result = 1 - pos_result

        logits = tf.concat([pos_result, neg_result], axis=1)

        self.train(logits)

    def train(self, logits):
        y = tf.one_hot(self.y, bimpm_args.class_size)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
        self.loss = tf.reduce_mean(loss)
        self.train_op = tf.train.AdamOptimizer(bimpm_args.learning_rate).minimize(self.loss)
        prediction = tf.argmax(logits, axis=1)
        correct_prediction = tf.equal(tf.cast(prediction, tf.int32), self.y)
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
