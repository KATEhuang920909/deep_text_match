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
    def __init__(self, embedding=None):
        self.q_w = tf.placeholder(dtype=tf.int32, shape=(None, bimpm_args.max_word_len), name="q_w")
        self.q_c = tf.placeholder(dtype=tf.int32, shape=(None, bimpm_args.max_char_len), name="q_c")
        self.d_w = tf.placeholder(dtype=tf.int32, shape=(None, bimpm_args.max_word_len), name="d_w")
        self.d_c = tf.placeholder(dtype=tf.int32, shape=(None, bimpm_args.max_char_len), name="d_c")
        self.y = tf.placeholder(dtype=tf.int32, shape=None, name="y")

        self.c_embedding = tf.get_variable(dtype=tf.float32,
                                           shape=(bimpm_args.char_vocab_len, bimpm_args.char_embedding_len),
                                           trainable=False,
                                           name="c_embedding")
        self.w_embedding = tf.get_variable(dtype=tf.float32,
                                           initializer=embedding,
                                           trainable=False,
                                           name="w_embedding")

        self.forward()

    # def dropout(self, x):
    #     return tf.nn.dropout(x, keep_prob=0.9)
    def cosine(self, p, h, axis=2, keep_dims=False, W=None):
        if W:
            p = tf.matmul(W, tf.transpose(p))
            h = tf.matmul(W, tf.transpose(h))

        if p.shape.as_list() != h.shape.as_list():
            p = tf.expand_dims(p, axis=0)
            # print("index")
        # print("p,h", p.shape, h.shape)
        p_norm = tf.norm(p, axis=axis, keep_dims=keep_dims)
        h_norm = tf.norm(h, axis=axis, keep_dims=keep_dims)
        cosine = tf.reduce_sum(tf.multiply(p, h), axis=axis, keep_dims=keep_dims) / (p_norm * h_norm)
        # print("cosine_shape", cosine.shape)
        return cosine

    def lstm(self, x):
        cell = tf.nn.rnn_cell.BasicLSTMCell(bimpm_args.hidden_size)
        return tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

    def bilstm(self, x):
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(bimpm_args.hidden_size)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(bimpm_args.hidden_size)
        return tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, dtype=tf.float32)

    def match_function(self, x, y, W, dims=1):
        """
        one vs one
        this function is f_m function,from paper.x and y is vectors and  W is trainable
        @param x: vector 1 shape  [batch_size,d]
        @param y: vector 2 shape  [batch_size,d]
        @param W: [perspective_size, d] there we define max_char_length as perspective-size
        """
        # print(x.shape, y.shape)
        x, y = tf.multiply(tf.expand_dims(x, dims), W), tf.multiply(tf.expand_dims(y, dims), W)  # batch_size,p_size,d
        #
        result = self.cosine(x, y, axis=-1, keep_dims=False)
        # print("result", result.shape)
        return result
        # [batch_size,p_size]

    def match_pooling_function(self, x, y, W):
        """
        more vs more
        this function is f_m function,from paper.x and y is vectors and  W is trainable
        @param x: vector 1 shape  [batch_size,d]
        @param y: vector 2 shape  [seq,batch_size,d]
        @param W: [perspective_size, d] there we define max_char_length as perspective-size
        """

        x, y = tf.multiply(tf.expand_dims(x, 1), W), tf.multiply(tf.expand_dims(y, 2), W)  # batch_size,p_size,d
        # print(x.shape, y.shape)
        result = self.cosine(x, y, axis=3, keep_dims=True)
        # result = tf.reshape(result, [result[0], result[1], result[2]])

        return tf.transpose(result, [1, 0, 2, 3])
        # [batch_size,len,len,p_size]

    def max_value_matching(self, x):
        """

        @param x:x shape :[15]
        @return:
        """
        result = []
        for num in x:
            result.append(x[:, num, :])
        return

    def forward(self):
        # step1: word_embedding
        q_w_embedding = tf.nn.embedding_lookup(self.w_embedding, self.q_w)
        # print(q_w_embedding.shape)
        d_w_embedding = tf.nn.embedding_lookup(self.w_embedding, self.d_w)
        # char_embedding
        q_c_embedding = tf.nn.embedding_lookup(self.c_embedding, self.q_c)
        d_c_embedding = tf.nn.embedding_lookup(self.c_embedding, self.d_c)

        # step2 input char_embedding to lstm:
        with tf.variable_scope("lstm_q_c"):
            q_c_output, _ = self.lstm(q_c_embedding)
            # print(q_c_output.shape)
        with tf.variable_scope("lstm_d_c"):
            d_c_output, _ = self.lstm(d_c_embedding)

        # step3 concat char_embedding and word_embedding
        q_embedding = tf.concat([q_w_embedding, q_c_output], axis=2)
        d_embedding = tf.concat([d_w_embedding, d_c_output], axis=2)
        # print("d_embedding", d_embedding.shape)
        q_embedding = tf.nn.dropout(q_embedding, keep_prob=0.9)
        d_embedding = tf.nn.dropout(d_embedding, keep_prob=0.9)
        # input embedidng to bilstm
        with tf.variable_scope("bilstm_q"):
            q_output, _ = self.bilstm(q_embedding)
            # print("q_output", q_output[0].shape)
        with tf.variable_scope("bilstm_d"):
            d_output, _ = self.bilstm(d_embedding)
        q_output = (tf.nn.dropout(q_output[0], keep_prob=0.9), tf.nn.dropout(q_output[1], keep_prob=0.9))
        d_output = (tf.nn.dropout(d_output[0], keep_prob=0.9), tf.nn.dropout(d_output[1], keep_prob=0.9))
        ##matching layer
        d = bimpm_args.hidden_size
        # full match
        W_1 = tf.Variable(initial_value=tf.truncated_normal([bimpm_args.max_char_len * 10, d]),
                          dtype=tf.float32,
                          name="W_1",
                          trainable=True)
        W_2 = tf.Variable(initial_value=tf.truncated_normal([bimpm_args.max_char_len * 10, d]),
                          dtype=tf.float32,
                          name="W_2",
                          trainable=True)

        q_f_output_tans = tf.transpose(q_output[0], [1, 0, 2])  # [len,batch_size,d]
        # print("q_f_output_tans", q_f_output_tans.shape)
        # print("d_output[0][-1]", d_output[0][:, -1, :].shape)
        q_b_output_tans = tf.transpose(q_output[-1], [1, 0, 2])
        d_f_output_tans = tf.transpose(d_output[0], [1, 0, 2])
        d_b_output_tans = tf.transpose(q_output[-1], [1, 0, 2])
        # print("q_f_output_tans", q_f_output_tans.shape)
        # output_size : len,batch_size,p_size
        q_f_full_output = tf.map_fn(fn=lambda x: self.match_function(d_output[0][:, -1, :], x, W_1),
                                    elems=q_f_output_tans)
        q_b_full_output = tf.map_fn(fn=lambda x: self.match_function(d_output[0][:, 0, :], x, W_2),
                                    elems=q_b_output_tans)
        d_f_full_output = tf.map_fn(fn=lambda x: self.match_function(q_output[0][:, -1, :], x, W_1),
                                    elems=d_f_output_tans)
        d_b_full_output = tf.map_fn(fn=lambda x: self.match_function(q_output[0][:, 0, :], x, W_2),
                                    elems=d_b_output_tans)
        q_f_full_output = tf.transpose(q_f_full_output, [1, 0, 2])
        q_b_full_output = tf.transpose(q_b_full_output, [1, 0, 2])
        d_f_full_output = tf.transpose(d_f_full_output, [1, 0, 2])
        d_b_full_output = tf.transpose(d_b_full_output, [1, 0, 2])
        # output_size : batch_size,len,p_size
        # print("d_b_full_output", d_b_full_output.shape)
        # maxpooling match
        W_3 = tf.Variable(initial_value=tf.truncated_normal([bimpm_args.max_char_len * 10, d]),
                          dtype=tf.float32,
                          name="W_3",
                          trainable=True)
        W_4 = tf.Variable(initial_value=tf.truncated_normal([bimpm_args.max_char_len * 10, d]),
                          dtype=tf.float32,
                          name="W_4",
                          trainable=True)
        q_f_maxpooling_output = tf.reduce_max(
            tf.map_fn(lambda x: self.match_pooling_function(x, d_f_output_tans, W_3), elems=q_f_output_tans),
            axis=2)
        q_b_maxpooling_output = tf.reduce_max(
            tf.map_fn(lambda x: self.match_pooling_function(x, d_b_output_tans, W_4), elems=q_b_output_tans),
            axis=2)
        d_f_maxpooling_output = tf.reduce_max(
            tf.map_fn(lambda x: self.match_pooling_function(x, q_f_output_tans, W_3), elems=d_f_output_tans),
            axis=2)
        d_b_maxpooling_output = tf.reduce_max(
            tf.map_fn(lambda x: self.match_pooling_function(x, q_b_output_tans, W_4), elems=d_b_output_tans),
            axis=2)

        # shape : batch_szie,p_size
        q_f_maxpooling_output = tf.reduce_sum(tf.transpose(q_f_maxpooling_output, [1, 0, 2, 3]), axis=-1)
        q_b_maxpooling_output = tf.reduce_sum(tf.transpose(q_b_maxpooling_output, [1, 0, 2, 3]), axis=-1)
        d_f_maxpooling_output = tf.reduce_sum(tf.transpose(d_f_maxpooling_output, [1, 0, 2, 3]), axis=-1)
        d_b_maxpooling_output = tf.reduce_sum(tf.transpose(d_b_maxpooling_output, [1, 0, 2, 3]), axis=-1)
        #  output_size :batch_size,len,p_size
        # print("d_b_maxpooling_output", d_b_maxpooling_output.shape)
        # attentive match
        # cosine:
        q_f_alpha = tf.map_fn(fn=lambda x: self.cosine(x, d_f_output_tans, axis=-1), elems=q_f_output_tans)
        q_b_alpha = tf.map_fn(fn=lambda x: self.cosine(x, d_b_output_tans, axis=-1), elems=q_b_output_tans)
        d_f_alpha = tf.map_fn(fn=lambda x: self.cosine(x, q_f_output_tans, axis=-1), elems=d_f_output_tans)
        d_b_alpha = tf.map_fn(fn=lambda x: self.cosine(x, q_b_output_tans, axis=-1), elems=d_b_output_tans)

        q_f_alpha = tf.transpose(q_f_alpha, [2, 0, 1])
        q_b_alpha = tf.transpose(q_b_alpha, [2, 0, 1])
        d_f_alpha = tf.transpose(d_f_alpha, [2, 0, 1])
        d_b_alpha = tf.transpose(d_b_alpha, [2, 0, 1])
        # print("d_b_alpha", d_b_alpha.shape)
        # alpha shape should be like: batch_size,query_shape,document_shape

        q_f_ = tf.matmul(q_f_alpha, d_output[0])
        q_b_ = tf.matmul(q_b_alpha, d_output[-1])
        d_f_ = tf.matmul(d_f_alpha, q_output[0])
        d_b_ = tf.matmul(d_b_alpha, d_output[-1])
        # print("q_f_", q_f_.shape)
        q_f_mean = tf.divide(q_f_, tf.reduce_sum(q_f_alpha, axis=2, keep_dims=True))
        q_b_mean = tf.divide(q_b_, tf.reduce_sum(q_b_alpha, axis=2, keep_dims=True))
        d_f_mean = tf.divide(d_f_, tf.reduce_sum(d_f_alpha, axis=2, keep_dims=True))
        d_b_mean = tf.divide(d_b_, tf.reduce_sum(d_b_alpha, axis=2, keep_dims=True))
        # print("q_f_mean", q_f_mean.shape)
        # mean shape should be like :
        W_5 = tf.Variable(initial_value=tf.truncated_normal([bimpm_args.max_char_len * 10, d]),
                          dtype=tf.float32,
                          name="W_5",
                          trainable=True)
        W_6 = tf.Variable(initial_value=tf.truncated_normal([bimpm_args.max_char_len * 10, d]),
                          dtype=tf.float32,
                          name="W_6",
                          trainable=True)
        q_f_att_output = self.match_function(q_f_mean, q_output[0], W_5, dims=2)

        q_b_att_output = self.match_function(q_b_mean, q_output[-1], W_6, dims=2)
        d_f_att_output = self.match_function(d_f_mean, d_output[0], W_5, dims=2)
        d_b_att_output = self.match_function(d_b_mean, d_output[-1], W_5, dims=2)
        # print("q_f_att_output", q_f_att_output.shape)

        # max attentive matching
        # highest cosine similarity as attentive vector

        # 4、Max-Attentive-Matching
        W_7 = tf.Variable(initial_value=tf.truncated_normal([bimpm_args.max_char_len * 10, d]),
                          dtype=tf.float32,
                          name="W_7",
                          trainable=True)
        W_8 = tf.Variable(initial_value=tf.truncated_normal([bimpm_args.max_char_len * 10, d]),
                          dtype=tf.float32,
                          name="W_8",
                          trainable=True)
        q_max_fw = tf.reduce_max(q_f_, axis=1, keep_dims=False)
        q_max_bw = tf.reduce_max(q_b_, axis=1, keep_dims=False)
        d_max_fw = tf.reduce_max(d_f_, axis=1, keep_dims=False)
        d_max_bw = tf.reduce_max(d_b_, axis=1, keep_dims=False)
        # print("d_max_bw", d_max_bw.shape)
        q_f_max_att = tf.map_fn(fn=lambda x: self.match_function(x, q_max_fw, W_7, dims=1), elems=q_f_output_tans)
        q_b_max_att = tf.map_fn(fn=lambda x: self.match_function(x, q_max_bw, W_8, dims=1), elems=q_b_output_tans)
        d_f_max_att = tf.map_fn(fn=lambda x: self.match_function(x, d_max_fw, W_7, dims=1), elems=d_f_output_tans)
        d_b_max_att = tf.map_fn(fn=lambda x: self.match_function(x, d_max_bw, W_8, dims=1), elems=d_b_output_tans)
        # print("d_b_max_att", d_b_max_att.shape)
        # q_f_alpha_max = tf.argmax(q_f_alpha, axis=2)  # batch_size ,query_size
        # q_b_alpha_max = tf.argmax(q_b_alpha, axis=2)
        # d_f_alpha_max = tf.argmax(d_f_alpha, axis=2)
        # d_b_alpha_max = tf.argmax(d_b_alpha, axis=2)
        # print("d_b_alpha_max", d_b_alpha_max.shape)
        # q_f_max_att = tf.map_fn(fn=lambda x: d_output[0][:, x, :],elems=q_f_alpha_max)  # batch_size,len,d
        # q_b_max_att = tf.map_fn(fn=lambda x: d_output[-1][:, x, :], elems=q_b_alpha_max)
        # d_f_max_att = tf.map_fn(fn=lambda x: q_output[0][:, x, :], elems=d_f_alpha_max)
        # d_b_max_att = tf.map_fn(fn=lambda x: q_output[-1][:, x, :], elems=d_b_alpha_max)
        # print("d_b_max_att", d_b_max_att.shape)
        q_f_max_att_output = tf.transpose(q_f_max_att, [1, 0, 2])  # len,batch_size,d
        q_b_max_att_output = tf.transpose(q_b_max_att, [1, 0, 2])  # len,batch_size,d
        d_f_max_att_output = tf.transpose(d_f_max_att, [1, 0, 2])  # len,batch_size,d
        d_b_max_att_output = tf.transpose(d_b_max_att, [1, 0, 2])  # len,batch_size,d
        # print("d_b_max_att", d_b_max_att.shape)
        # batch_size ,query_size
        # q_f_max_att_output = self.match_function(q_f_output_tans, q_f_max_att,W_)
        # q_b_max_att_output = self.match_function(q_f_output_tans, q_b_max_att)
        # d_f_max_att_output = self.match_function(d_f_output_tans, d_f_max_att)
        # d_b_max_att_output = self.match_function(d_b_output_tans, d_b_max_att)
        # q_f_max_att_output = tf.transpose(q_f_max_att_output, [1, 0, 2])  # len,batch_size,d
        # q_b_max_att_output = tf.transpose(q_b_max_att_output, [1, 0, 2])  # len,batch_size,d
        # d_f_max_att_output = tf.transpose(d_f_max_att_output, [1, 0, 2])  # len,batch_size,d
        # d_b_max_att_output = tf.transpose(d_b_max_att_output, [1, 0, 2])  # len,batch_size,d
        #  output_size :batch_size,len,p_size
        # concat
        q_final_output = tf.concat(
            (q_f_full_output, q_f_maxpooling_output, q_f_att_output, q_f_max_att_output,
             q_b_full_output, q_b_maxpooling_output, q_b_att_output, q_b_max_att_output),
            axis=2)
        # batch_size ,len ,p_size*8
        d_final_output = tf.concat(
            (d_f_full_output, d_f_maxpooling_output, d_f_att_output, d_f_max_att_output,
             d_b_full_output, d_b_maxpooling_output, d_b_att_output, d_b_max_att_output),
            axis=2)

        # ----- Aggregation Layer -----
        with tf.variable_scope("bilstm_agg_p", reuse=tf.AUTO_REUSE):
            (q_f_last, q_b_last), _ = self.bilstm(q_final_output)
        with tf.variable_scope("bilstm_agg_h", reuse=tf.AUTO_REUSE):
            (d_f_last, d_b_last), _ = self.bilstm(d_final_output)

        x = tf.concat((q_f_last, q_b_last, d_f_last, d_b_last), axis=2)
        x = tf.reshape(x, shape=[-1, x.shape[1] * x.shape[2]])
        # batch_size ,len*hidden_size*4
        x = tf.nn.dropout(x, keep_prob=0.9)
        # ----- Prediction Layer -----
        # x = tf.layers.dense(x, 20000, activation='relu')
        # x = self.dropout(x)
        x = tf.layers.dense(x, 1024, activation='tanh')
        x = tf.nn.dropout(x, keep_prob=0.9)
        x = tf.layers.dense(x, 512)
        x = tf.nn.dropout(x, keep_prob=0.9)
        # x = self.dropout(x)
        logits = tf.layers.dense(x, bimpm_args.class_size)

        self.train(logits)

    def train(self, logits):
        y = tf.one_hot(self.y, bimpm_args.class_size)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
        self.loss = tf.reduce_mean(loss)
        self.train_op = tf.train.AdamOptimizer(bimpm_args.learning_rate).minimize(self.loss)
        prediction = tf.argmax(logits, axis=1)
        correct_prediction = tf.equal(tf.cast(prediction, tf.int32), self.y)
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
