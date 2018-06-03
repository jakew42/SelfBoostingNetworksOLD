import argparse
from functools import reduce

import numpy as np
import sonnet as snt
import tensorflow as tf

import boosted_classifier
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
classifier = boosted_classifier.NaiveBoostedClassifier(3, label_shape[1])

# build model
data_ph = tf.placeholder(tf.float32, shape=data_shape)
label_ph = tf.placeholder(tf.int32, shape=label_shape)  # should be one-hot
final_classification, weak_logits = classifier(data_ph)
weak_classifications = [tf.nn.softmax(logits) for logits in weak_logits]

correct_prop = tf.count_nonzero(
    tf.equal(tf.argmax(final_classification, 1), tf.argmax(label_ph, 1)),
    dtype=tf.float32) / tf.constant(
        args.batch_size, dtype=tf.float32)


def feed_dict_fn():
    data, labels = next(train_gen)
    return {data_ph: data, label_ph: labels}


# calculate boosting weights
weights = tf.constant(
    1. / args.batch_size, dtype=tf.float32, shape=(args.batch_size, ))
losses = []
for i, classification in enumerate(weak_classifications):
    scale = -(float(class_num) - 1.) / float(class_num)
    weights = weights * tf.reduce_sum(
        tf.exp(scale *
               (tf.to_float(label_ph) * tf.log(classification + 1e-10))),
        axis=1)
    losses.append(weights * tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.argmax(label_ph, axis=1), logits=weak_logits[i]))

# calculate gradients
optimizer = tf.train.AdamOptimizer()
grads_and_vars = [optimizer.compute_gradients(l) for l in losses]
vars = list(zip(*grads_and_vars[0]))[1]
grads = [[v if v is not None else 0 for v in list(zip(*gv))[0]]
         for gv in grads_and_vars]
out_grads = tf.global_norm([g for g in grads[0] if g != 0])
grads = [sum(gs) for gs in list(zip(*grads))]
update = list(zip(grads, vars))
capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in update]
train_op = optimizer.apply_gradients(capped_gvs)

# initialize session and train
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as session:
    session.run(tf.global_variables_initializer())
    for epoch in range(args.epochs):
        print("EPOCH {}".format(epoch))
        norms, acc = util.run_epoch_ops(
            session,
            train_data.shape[0] // args.batch_size,
            silent_ops=[train_op],
            verbose_ops=[out_grads, correct_prop],
            feed_dict_fn=feed_dict_fn,
            verbose=True)
        print(norms)
        print(acc)
