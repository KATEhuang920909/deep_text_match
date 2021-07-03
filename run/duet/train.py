import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.logger_config import logger
from models.duet import Graph
import tensorflow as tf
from utils.load_data import load_char_data
from args import duet_args

p, h, y = load_char_data('data/train.csv', data_size=None)
p_eval, h_eval, y_eval = load_char_data('data/dev.csv', data_size=None)

p_holder = tf.placeholder(dtype=tf.int32, shape=(None, duet_args.seq_length), name='p')
h_holder = tf.placeholder(dtype=tf.int32, shape=(None, duet_args.seq_length), name='h')
y_holder = tf.placeholder(dtype=tf.int32, shape=None, name='y')
p_eval_holder = tf.placeholder(dtype=tf.int32, shape=(None, duet_args.seq_length), name='p')
h_eval_holder = tf.placeholder(dtype=tf.int32, shape=(None, duet_args.seq_length), name='h')
y_eval_holder = tf.placeholder(dtype=tf.int32, shape=None, name='y')

dataset = tf.data.Dataset.from_tensor_slices((p_holder, h_holder, y_holder))
dataset = dataset.batch(duet_args.batch_size).repeat(duet_args.epochs)
iterator_train = dataset.make_initializable_iterator()
next_element_train = iterator_train.get_next()
iterator_eval = dataset.make_initializable_iterator()
next_element_eval = iterator_eval.get_next()

model = Graph()
saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9

with tf.Session(config=config)as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(r'../../output/duet', sess.graph)
    sess.run(iterator_train.initializer, feed_dict={p_holder: p,
                                                    h_holder: h,
                                                    y_holder: y})
    sess.run(iterator_eval.initializer, feed_dict={p_holder: p_eval,
                                                   h_holder: h_eval,
                                                   y_holder: y_eval})
    steps = int(len(y) / duet_args.batch_size)
    for epoch in range(duet_args.epochs):
        for step in range(steps):
            p_batch, h_batch, y_batch = sess.run(next_element_train)
            _, loss, acc = sess.run([model.train_op, model.loss, model.acc],
                                    feed_dict={model.q: p_batch,
                                               model.d: h_batch,
                                               model.y: y_batch,
                                               model.keep_prob: duet_args.keep_prob})
            logger.info(f'epoch:, {epoch},  step:, {step},  loss: , {loss},  acc:, {acc}')
        step_eval = int(len(y_eval) / duet_args.batch_size)
        acc_final = 0
        loss_final = 0
        for step in range(step_eval):
            p_batch, h_batch, y_batch = sess.run(next_element_eval)
            loss_eval, acc_eval = sess.run([model.loss, model.acc],
                                           feed_dict={model.q: p_batch,
                                                      model.d: h_batch,
                                                      model.y: y_batch,
                                                      model.keep_prob: 1})
            acc_final += acc_eval

            loss_final += loss_eval

            logger.info(f'loss_eval:{loss_eval},  acc_eval: {acc_eval}')
        acc_final = acc_final / step_eval
        loss_final = loss_final / step_eval
        logger.info(f'loss_eval_final:{loss_final},  acc_eval_final: {acc_final}')
        print('\n')
        saver.save(sess, f'../../output/duet/duet_{epoch}.ckpt')

