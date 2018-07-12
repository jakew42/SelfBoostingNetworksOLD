import argparse
from functools import reduce

import numpy as np
import sonnet as snt
import tensorflow as tf

import networks.stems
import networks.classifiers
import data.data as data
import util

parser = argparse.ArgumentParser(description='Self-Boosting Network Training.')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--blocks', type=int, default=5)
args = parser.parse_args()

# load the data
train_data, train_labels, _, _ = data.load_cifar10('./data/')
data_shape = (args.batch_size, ) + train_data.shape[1:]
label_shape = (args.batch_size, ) + train_labels.shape[1:]
class_num = train_labels.shape[1]
train_gen = data.parallel_data_generator([train_data, train_labels],
                                         args.batch_size)

# define the model
stem = networks.stems.BigConvStem(name='stem')
classifier = networks.classifiers.ReduceFlattenClassifier(class_num)

# build model
data_ph = tf.placeholder(tf.float32, shape=data_shape)
label_ph = tf.placeholder(tf.int32, shape=label_shape)  # should be one-hot
stem_repr = stem(data_ph)
final_classification = classifier(stem_repr)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=tf.argmax(label_ph, axis=1), logits=final_classification)

class_rate_fn = lambda a: tf.count_nonzero(
    tf.equal(tf.argmax(a, 1), tf.argmax(label_ph, 1)),
    dtype=tf.float32) / tf.constant(
        args.batch_size, dtype=tf.float32)
correct_final_prop = class_rate_fn(final_classification)

print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='stem'))
saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='stem'))

def feed_dict_fn():
    data, labels = next(train_gen)
    return {data_ph: data, label_ph: labels}

# calculate gradients
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss)

print("Trainable Parameters: {}".format(
    np.sum(
        [np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

# initialize session and train
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as session:
    session.run(tf.global_variables_initializer())
    for epoch in range(args.epochs):
        print("EPOCH {}".format(epoch))
        out_dict = util.run_epoch_ops(
            session,
            train_data.shape[0] // args.batch_size,
            silent_ops=[train_op],
            verbose_ops={'accuracy': correct_final_prop},
            feed_dict_fn=feed_dict_fn,
            verbose=True)
        print("Accuracy: " + str(np.mean(out_dict['accuracy'])))
    saver.save(session, './BigConvStem')
