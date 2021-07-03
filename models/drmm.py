# -*- coding: utf-8 -*-
"""
 Time : 2021/6/30 19:26
 Author : huangkai
 File : siamese.py
 mail:18707125049@163.com
 paper:https://arxiv.org/pdf/1711.08611.pdf
"""
import tensorflow as tf
from args import drmm_args
import numpy as np
from tensorflow.contrib.layers.python.ops import bucketization_op
from collections import Counter


class Graph:
    def __init__(self, embedding_type="ONE_HOT", embedding=None):
        self.q = tf.placeholder(dtype=tf.int32, shape=(None, drmm_args.query_seq_length), name='q')
        self.d = tf.placeholder(dtype=tf.int32, shape=(None, drmm_args.document_seq_length), name='d')
        self.y = tf.placeholder(dtype=tf.int32, shape=None, name='y')

        self.keep_prob = tf.placeholder(dtype=tf.float32, name='drop_rate')
        #
        if embedding_type == "ONE_HOT":
            self.embedding = tf.get_variable(dtype=tf.float32,
                                             shape=(drmm_args.vocab_size, drmm_args.char_embedding_size),
                                             name='embedding',
                                             trainable=True)
        elif embedding_type == "WORD2VEC":
            self.embedding = tf.get_variable(dtype=tf.float32, initializer=embedding, trainable=False)
        self.forward()

    def tensor2bucket(self, x, bins):

        length = len(bins)
        ls = [0] * length
        for i in range(length):
            ls[i] = tf.reduce_sum(tf.cast(tf.equal(x, i), tf.float32))
        return tf.cast(tf.convert_to_tensor(ls), tf.float32)

    def bucket_histogram(self, VECTOR, histogram_type="CH", sim_type="cos"):
        """

        Args:
            VECTOR:input array for calcul
            type:the type for building histogram we have 3 methods==================================

            1、"CH","Count-based Histogram"：this is the simplest way to build histogram vec，it counts interaction's bin
             as histogram's value,for example:
            cos similarity ranges [-1,1],so we have five bins:{[-1,-0.5],[-0.5,0],[0,0.5],[0.5,1]},the one word from
            query to document's interaction similarity vector we have [1,0.2,0.7,0.3,-0.1,-0.2,-0.3],and the histogram
            value is [2,3,2,0],the vector's length is the num of bins

            2、"NH","Normalized Histogram": we use normalize operation based on "CH",thus focus on relative value instead
            of absolute value
            3、"LCH","LogCount-based Histogram":we apply log function based on "CH",thus more easily to learn multi
            relationships.

        Returns: histogram vec

        """
        # vector = {'vector': VECTOR}
        # print("buckets shape", VECTOR.shape)
        # vector_fc = tf.feature_column.numeric_column('vector')
        #
        bins = list(np.arange(-1, 1, 0.1))
        buckets = []
        # buckets = tf.feature_column.bucketized_column(VECTOR, boundaries=bins)
        vec = bucketization_op.bucketize(VECTOR, boundaries=bins)
        # print("vec shape",vec.shape)
        buckets = tf.map_fn(fn=lambda x: self.tensor2bucket(x, bins=bins), elems=vec, dtype=tf.float32)
        # for array in vec2array:
        #     ls = [0] * len(bins)
        #     for unit in array:
        #         ls[unit] += 1
        #     buckets.append(ls)
        # buckets = np.array(buckets)
        # buckets = tf.feature_column.input_layer(vector, [buckets])
        print("buckets shape2:", buckets.shape)
        # buckets = tf.reduce_sum(buckets, axis=0)
        # print("buckets shape2:", buckets.shape)
        if histogram_type == "CH":
            return buckets
        if histogram_type == "NH":
            return tf.norm(buckets, axis=2)
        if histogram_type == "LCH":
            return tf.log(buckets)

    def gate_network(self, x):
        """

        Args:
            x:x.shape==>[batch_size,seq_len,vec_size]
            1.term vector :x verctor can be term vector such as word2vec 、glove 、elmo etc
            2.idf :we can use idf static feature as input to gating function

        Returns:weght_value: [batch_size,seq_len,1]

        """
        # weights = tf.get_variable(name="gate_weight", dtype=tf.float32, shape=[x.shape[2], 1],
        #                           initial_value=tf.glorot_uniform_initializer())
        # bias = tf.get_variable(name="gate_bias", dtype=tf.float32, shape=[-1, 1],
        #                        initializer=tf.random_uniform_initializer)
        # output = tf.add(tf.matmul(x, weights), bias)
        x = tf.layers.batch_normalization(x)
        output = tf.layers.dense(x, 1, activation="tanh")

        return output

    def fully_connect(self, x):
        x = tf.layers.batch_normalization(x)
        x = tf.layers.dense(x, 30, activation='tanh')
        x = tf.layers.batch_normalization(x)
        x = tf.layers.dense(x, 5, activation='tanh')
        x = tf.layers.batch_normalization(x)
        x = tf.layers.dense(x, 1, activation='tanh')

        return x

    # def hinge_loss(self, x, y):
    #     # temp1 = y * x
    #     # temp2 = (1 - y) * x
    #     loss = tf.maximum(0.0, 1.0 - (x*y))
    #     return loss
    @staticmethod
    def cosine(p, h):
        p_norm = tf.norm(p, axis=1, keepdims=True)
        h_norm = tf.norm(h, axis=1, keepdims=True)

        cosine = tf.reduce_sum(tf.multiply(p, h), axis=1, keepdims=True) / (p_norm * h_norm)

        return cosine

    def forward(self):
        q_embedding = tf.nn.embedding_lookup(self.embedding, self.q)
        d_embedding = tf.nn.embedding_lookup(self.embedding, self.d)
        # local interaction
        # use cos function
        output_1 = tf.norm(q_embedding, axis=2, keep_dims=True) * tf.norm(d_embedding, axis=2, keep_dims=True)
        print(tf.norm(q_embedding, axis=2, keep_dims=True).shape)
        print(tf.norm(d_embedding, axis=2, keep_dims=True).shape)
        d_embedding = tf.transpose(d_embedding, [0, 2, 1])
        output_2 = tf.matmul(q_embedding, d_embedding)

        output = output_2 / output_1
        print("cos shape", output.shape)
        output = tf.transpose(output, [1, 0, 2])
        print("before histogram mapping:", output.shape)

        # histogram mapping
        output = tf.map_fn(fn=lambda x: self.bucket_histogram(x, histogram_type="CH", sim_type="cos"),
                           elems=output,
                           name="histogram_mapping")
        print("histogram mapping:", output.shape)

        # output shape : (batch_size,seq_len,bins)
        output = tf.map_fn(fn=lambda x: self.fully_connect(x),
                           elems=output,
                           name="full_connected")
        print("outputshaoe0:", output.shape)
        output = tf.transpose(output, [1, 0, 2])
        # output shape :(batch_size,seq_len,1)
        print("outputshaoe:", output.shape)
        # gate network
        q_embedding_gate = tf.transpose(q_embedding, [1, 0, 2])
        term_gating = tf.map_fn(fn=lambda x: self.gate_network(x),
                                elems=q_embedding_gate,
                                name="term_gate")
        term_gating = tf.nn.softmax(term_gating)
        print("term_gating:", term_gating.shape)
        term_gating = tf.transpose(term_gating, [1, 0, 2])

        # gate shape: (batch_size,seq_len,1)
        pos_result = tf.reduce_sum(tf.multiply(term_gating, output), axis=1)
        print("pos_result:", pos_result.shape)
        pos_result = tf.reshape(pos_result, [-1, 1])

        # (batch,score)

        # add
        neg_result = 1 - pos_result

        logits = tf.concat([pos_result, neg_result], axis=1)

        self.train(logits)

    def train(self, logits):
        y = tf.one_hot(self.y, drmm_args.class_size)
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
        self.loss = tf.reduce_mean(loss)
        self.train_op = tf.train.AdamOptimizer(drmm_args.learning_rate).minimize(self.loss)
        prediction = tf.argmax(logits, axis=1)
        correct_prediction = tf.equal(tf.cast(prediction, tf.int32), self.y)
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
