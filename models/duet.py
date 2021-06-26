# -*- coding: utf-8 -*-
"""
@Time    : 2021/6/23 20:57
@Author  : huangkai21
@file    : duet.py
"""
import tensorflow as tf
from args import duet_args


class Graph:
    def __init__(self, embedding_type="ONE_HOT", embedding=None):
        self.q = tf.placeholder(dtype=tf.int32, shape=(None, duet_args.seq_length), name='q')
        self.d = tf.placeholder(dtype=tf.int32, shape=(None, duet_args.seq_length), name='d')
        self.y = tf.placeholder(dtype=tf.int32, shape=None, name='y')

        self.keep_prob = tf.placeholder(dtype=tf.float32, name='drop_rate')
        #
        if embedding_type == "ONE_HOT":
            self.embedding = tf.get_variable(dtype=tf.float32,
                                             shape=(duet_args.vocab_size, duet_args.char_embedding_size),
                                             name='embedding')
        elif embedding_type == "WORD2VEC":
            self.embedding = tf.get_variable(dtype=tf.float32, initializer=embedding, trainable=False)
        self.forward()

    def dropout(self, x):
        return tf.nn.dropout(x, keep_prob=self.keep_prob)

    def fully_connect(self, x):
        x = tf.layers.dense(x, 300, activation='tanh')
        x = self.dropout(x)
        x = tf.layers.dense(x, 300, activation='tanh')
        x = self.dropout(x)
        x = tf.layers.dense(x, 300, activation='tanh')
        x = self.dropout(x)
        x = tf.layers.dense(x, 1)

        return x

    @staticmethod
    def output(q, d):
        cosine = tf.add(q, d)

        return cosine

    def forward(self):
        q_embedding = tf.nn.embedding_lookup(self.embedding, self.q)
        d_embedding = tf.nn.embedding_lookup(self.embedding, self.d)
        # local model
        q_embedding1 = tf.transpose(q_embedding, [0, 2, 1])
        output1 = tf.matmul(d_embedding, q_embedding1)
        output1 = tf.layers.conv2d(tf.expand_dims(output1, axis=3),
                                   filters=duet_args.cnn1_filters,
                                   kernel_size=[duet_args.seq_length, duet_args.filter_width])

        output1 = tf.reshape(output1, shape=(-1, output1.shape[2] * output1.shape[3]))
        output1 = self.fully_connect(output1)
        # distributed model
        q_embedding2 = tf.layers.conv2d(tf.expand_dims(q_embedding, axis=3),
                                        filters=duet_args.cnn1_filters,
                                        kernel_size=[duet_args.filter_width2, duet_args.char_embedding_size])
        d_embedding2 = tf.layers.conv2d(tf.expand_dims(d_embedding, axis=3),
                                        filters=duet_args.cnn1_filters,
                                        kernel_size=[duet_args.filter_width2, duet_args.char_embedding_size])
        q_embedding2 = tf.transpose(q_embedding2, [0, 3, 1, 2])
        d_embedding2 = tf.transpose(d_embedding2, [0, 3, 1, 2])
        # 这里query和document截断长度相同，因此windows-based pooling采用了一个较小的窗口
        q_embedding2 = tf.layers.max_pooling2d(q_embedding2, strides=1, pool_size=[1, q_embedding2.shape[2]])

        d_embedding2 = tf.layers.max_pooling2d(d_embedding2, strides=1, pool_size=[1, d_embedding2.shape[2] // 5])

        q_embedding2 = tf.reshape(q_embedding2, shape=(-1, q_embedding2.shape[1] * q_embedding2.shape[2]))
        q_embedding2 = tf.layers.dense(q_embedding2, duet_args.cnn1_filters, activation='tanh')

        d_embedding2 = tf.layers.conv2d(d_embedding2,
                                        filters=duet_args.cnn1_filters,
                                        kernel_size=[duet_args.cnn1_filters, 1])
        q_embedding2 = tf.reshape(q_embedding2, shape=[-1, duet_args.cnn1_filters, 1, 1])
        d_embedding2 = tf.transpose(d_embedding2, [0, 3, 2, 1])

        output2 = tf.multiply(d_embedding2, q_embedding2)
        output2 = tf.reshape(output2, shape=(-1, output2.shape[1] * output2.shape[2]))
        output2 = self.fully_connect(output2)
        # add
        pos_result = tf.add(output1, output2)
        neg_result = 1 - pos_result

        logits = tf.concat([pos_result, neg_result], axis=1)

        self.train(logits)

    def train(self, logits):
        y = tf.one_hot(self.y, duet_args.class_size)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
        self.loss = tf.reduce_mean(loss)
        self.train_op = tf.train.AdamOptimizer(duet_args.learning_rate).minimize(self.loss)
        prediction = tf.argmax(logits, axis=1)
        correct_prediction = tf.equal(tf.cast(prediction, tf.int32), self.y)
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
