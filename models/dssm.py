import tensorflow as tf
from args import dssm_args


class Graph:
    def __init__(self, embedding_type="RANDOM_INIT", embedding=None):
        self.p = tf.placeholder(dtype=tf.int32, shape=(None, dssm_args.seq_length), name='p')
        self.h = tf.placeholder(dtype=tf.int32, shape=(None, dssm_args.seq_length), name='h')
        self.y = tf.placeholder(dtype=tf.int32, shape=None, name='y')

        self.keep_prob = tf.placeholder(dtype=tf.float32, name='drop_rate')
        #
        if embedding_type == "RANDOM_INIT":
            self.embedding = tf.get_variable(dtype=tf.float32,
                                             shape=(dssm_args.vocab_size, dssm_args.char_embedding_size),
                                             name='embedding')
        elif embedding_type == "WORD2VEC":
            self.embedding = tf.get_variable(dtype=tf.float32, initializer=embedding, trainable=False)
        self.forward()

    def dropout(self, x):
        return tf.nn.dropout(x, keep_prob=self.keep_prob)

    def fully_connect(self, x):
        x = tf.layers.dense(x, 256, activation='tanh')
        x = self.dropout(x)
        x = tf.layers.dense(x, 512, activation='tanh')
        x = self.dropout(x)
        x = tf.layers.dense(x, 256, activation='tanh')
        x = self.dropout(x)
        x = tf.reshape(x, shape=(-1, x.shape[1] * x.shape[2]))

        return x

    def cosine(self, p, h):
        p_norm = tf.norm(p, axis=1, keepdims=True)
        h_norm = tf.norm(h, axis=1, keepdims=True)

        cosine = tf.reduce_sum(tf.multiply(p, h), axis=1, keepdims=True) / (p_norm * h_norm)

        return cosine

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
        y = tf.one_hot(self.y, dssm_args.class_size)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
        self.loss = tf.reduce_mean(loss)
        self.train_op = tf.train.AdamOptimizer(dssm_args.learning_rate).minimize(self.loss)
        prediction = tf.argmax(logits, axis=1)
        correct_prediction = tf.equal(tf.cast(prediction, tf.int32), self.y)
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
