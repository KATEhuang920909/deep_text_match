import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from gensim.models.word2vec import Word2Vec
from utils.logger_config import logger
from models.bimpm_v2 import Graph
import tensorflow as tf
from utils.load_data import Data2idx
from args import bimpm_args

data2idx = Data2idx()
q, d, y = data2idx.load_data('data/train.csv')

q_eval, d_eval, y_eval = data2idx.load_data('data/dev.csv')
q_c_idx, d_c_idx = data2idx.load_char_idx(q, d, bimpm_args.max_char_len, bimpm_args.max_word_len)
q_eval_c_idx, d_eval_c_idx = data2idx.load_char_idx(q_eval, d_eval, bimpm_args.max_char_len, bimpm_args.max_word_len)
model = Word2Vec.load("../../output/word2vec/word2vec.model")
w2indx, w2vec = data2idx.create_dictionaries(model)
_, embedding = data2idx.get_embedding(w2indx, w2vec, bimpm_args.word_embedding_len)
embedding = tf.cast(embedding, dtype=tf.float32)
word_idx = data2idx.load_word_idx(w2indx, bimpm_args.max_word_len, q, d, q_eval, d_eval)
q_w_idx, d_w_idx, q_eval_w_idx, d_eval_w_idx = word_idx[0], word_idx[1], word_idx[2], word_idx[3]
q_c_index_holder = tf.placeholder(name='q_c_index', shape=(None, bimpm_args.max_char_len), dtype=tf.int32)
d_c_index_holder = tf.placeholder(name='d_c_index', shape=(None, bimpm_args.max_char_len), dtype=tf.int32)
q_w_index_holder = tf.placeholder(name='q_w_index', shape=(None, bimpm_args.max_word_len), dtype=tf.int32)
d_w_index_holder = tf.placeholder(name='d_w_index', shape=(None, bimpm_args.max_word_len), dtype=tf.int32)
label_holder = tf.placeholder(name='label', shape=(None,), dtype=tf.int32)

dataset = tf.data.Dataset.from_tensor_slices(
    (q_c_index_holder, d_c_index_holder, q_w_index_holder, d_w_index_holder, label_holder))
dataset = dataset.batch(bimpm_args.batch_size).repeat(bimpm_args.epochs)
iterator_train = dataset.make_initializable_iterator()
next_element_train = iterator_train.get_next()
iterator_eval = dataset.make_initializable_iterator()
next_element_eval = iterator_eval.get_next()
model = Graph(embedding=embedding)
saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7

with tf.Session(config=config)as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(r'../../output/bimpm', sess.graph)
    sess.run(iterator_train.initializer, feed_dict={q_c_index_holder: q_c_idx,
                                                    d_c_index_holder: d_c_idx,
                                                    q_w_index_holder: q_w_idx,
                                                    d_w_index_holder: d_w_idx,
                                                    label_holder: y})
    sess.run(iterator_eval.initializer, feed_dict={q_c_index_holder: q_eval_c_idx,
                                                   d_c_index_holder: d_eval_c_idx,
                                                   q_w_index_holder: q_eval_w_idx,
                                                   d_w_index_holder: d_eval_w_idx,
                                                   label_holder: y_eval})
    steps = int(len(y) / bimpm_args.batch_size)
    for epoch in range(bimpm_args.epochs):
        for step in range(steps):
            q_c_batch, d_c_batch, q_w_batch, d_w_batch, y_batch = sess.run(next_element_train)
            _, loss, acc = sess.run([model.train_op, model.loss, model.acc],
                                    feed_dict={model.q_c: q_c_batch,
                                               model.d_c: d_c_batch,
                                               model.q_w: q_w_batch,
                                               model.d_w: d_w_batch,
                                               model.y: y_batch})
            logger.info(f'epoch:, {epoch},  step:, {step},  loss: , {loss},  acc:, {acc}')
        step_eval = int(len(y_eval) / bimpm_args.batch_size)
        acc_final = 0
        loss_final = 0
        for step in range(step_eval):
            q_c_batch, d_c_batch, q_w_batch, d_w_batch, y_batch = sess.run(next_element_eval)
            loss_eval, acc_eval = sess.run([model.loss, model.acc],
                                           feed_dict={model.q_c: q_c_batch,
                                                      model.d_c: d_c_batch,
                                                      model.q_w: q_w_batch,
                                                      model.d_w: d_w_batch,
                                                      model.y: y_batch})
            acc_final += acc_eval

            loss_final += loss_eval

            logger.info(f'loss_eval:{loss_eval},  acc_eval: {acc_eval}')
        acc_final = acc_final / step_eval
        loss_final = loss_final / step_eval
        logger.info(f'loss_eval_final:{loss_final},  acc_eval_final: {acc_final}')
        print('\n')
        saver.save(sess, f'../../output/bimpm/bimpm_{epoch}.ckpt')
