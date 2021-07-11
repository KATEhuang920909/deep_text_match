import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.logger_config import logger
from models.siamese import Graph
import tensorflow as tf
from utils.load_data import Data2idx
from args import siamese_args
data2idx = Data2idx()
q, d, y = data2idx.load_data('data/train.csv', frac=0.05)

q_eval, d_eval, y_eval = data2idx.load_data('data/dev.csv', frac=0.05)
q_c_idx, d_c_idx = data2idx.load_char_idx(q, d, siamese_args.seq_length, siamese_args.seq_length)

q_holder = tf.placeholder(dtype=tf.int32, shape=(None, siamese_args.seq_length), name='p')
d_holder = tf.placeholder(dtype=tf.int32, shape=(None, siamese_args.seq_length), name='h')
y_holder = tf.placeholder(dtype=tf.int32, shape=None, name='y')

dataset = tf.data.Dataset.from_tensor_slices((q_holder, d_holder, y_holder))
dataset = dataset.batch(siamese_args.batch_size).repeat(siamese_args.epochs)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

model = Graph()
saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9

with tf.Session(config=config)as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(r'../../output/siamese', sess.graph)
    sess.run(iterator.initializer, feed_dict={q_holder: q_c_idx, d_holder: d_c_idx, y_holder: y})
    steps = int(len(y) / siamese_args.batch_size)
    for epoch in range(siamese_args.epochs):
        for step in range(steps):
            p_batch, h_batch, y_batch = sess.run(next_element)
            _, loss, acc = sess.run([model.train_op, model.loss, model.acc],
                                    feed_dict={model.q: p_batch,
                                               model.d: h_batch,
                                               model.y: y_batch,
                                               model.keep_prob: siamese_args.keep_prob})
            logger.info(f'epoch:, {epoch},  step:, {step},  loss: , {loss},  acc:, {acc}')

        loss_eval, acc_eval = sess.run([model.loss, model.acc],
                                       feed_dict={model.q: p_eval,
                                                  model.d: h_eval,
                                                  model.y: y_eval,
                                                  model.keep_prob: 1})

        logger.info(f'loss_eval:{loss_eval},  acc_eval: {acc_eval}')
        print('\n')
        saver.save(sess, f'../../output/siamese/siamese_{epoch}.ckpt')
