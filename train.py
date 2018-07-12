import argparse
import os
from functools import reduce

import numpy as np
import sonnet as snt
import tensorflow as tf

import boosted_classifier
import boosting_strategy
import data.data as data
import util
import voting_strategy
from networks import blocks, classifiers, stems

parser = argparse.ArgumentParser(description='Self-Boosting Network Training.')
parser.add_argument(
    '--classes', nargs='*', type=int, default=[], help='e.g., --classes 3 7 8')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--blocks', type=int, default=5)
parser.add_argument('--block_type', type=str, default='IdentityBlock')
parser.add_argument(
    '--classifier', type=str, default='ReduceFlattenClassifier')
parser.add_argument('--stem', type=str, default='BigConvStem')
parser.add_argument('--load_stem', action="store_true")
parser.add_argument('--voting_strategy', type=str, default='samme.r')
parser.add_argument('--boosting_strategy', type=str, default='samme.r')
parser.add_argument('--log_dir', type=str, default='./logs/')
metrics_args = parser.add_argument_group('Metrics')
metrics_args.add_argument('--gradient_norms', action="store_true")
metrics_args.add_argument('--weak_classifier_covar', action="store_true")
histogram_args = parser.add_argument_group('Plots')
histogram_args.add_argument('--weak_classifier_rate', action="store_true")
metrics_args.add_argument('--weak_classifier_conf_matrix', action="store_true")
args = parser.parse_args()

CIFAR10_CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
    'ship', 'truck'
]

# Setup log directory and log files
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
if args.gradient_norms:
    with open(os.path.join(args.log_dir, 'grad_norms.csv'), 'w') as csv:
        csv.write('pre_clip_grad_vals, post_clip_grad_vals,{}\n'.format(
            ','.join([
                'pre_clip_wc_{}_grad_vals'.format(i)
                for i in range(args.blocks)
            ])))
if args.weak_classifier_covar:
    with open(os.path.join(args.log_dir, 'weak_learner_covariance.csv'),
              'w') as csv:
        csv.write('weak_classifier_covariance')
if args.weak_classifier_conf_matrix and not os.path.exists(
        os.path.join(args.log_dir, 'wc_confusion_matrices')):
    os.makedirs(os.path.join(args.log_dir, 'wc_confusion_matrices'))

block_dict = {
    'IdentityBlock': blocks.IdentityBlock,
    'ResidualConvBlock': blocks.ResidualConvBlock
}
classifier_dict = {
    'ReduceFlattenClassifier': classifiers.ReduceFlattenClassifier
}
stem_dict = {'BigConvStem': stems.BigConvStem}

# load the data
train_data, train_labels, _, _ = data.load_cifar10('./data/')

# Use all classes if the default value of [] is present
if args.classes:
    dense_labels = np.argmax(train_labels, axis=1)
    idx_list = np.where(np.isin(dense_labels, args.classes))
    train_data = train_data[idx_list]
    train_labels = train_labels[idx_list]

data_shape = (args.batch_size, ) + train_data.shape[1:]
label_shape = (args.batch_size, ) + train_labels.shape[1:]
class_num = train_labels.shape[1]
train_gen = data.parallel_data_generator([train_data, train_labels],
                                         args.batch_size)

voting_strategy_dict = {
    'naive':
    voting_strategy.naive_voting_strategy,
    'linear_combo':
    voting_strategy.LinearComboStrategy(args.blocks),
    'mlp_combo':
    voting_strategy.MLPComboStrategy([50, 20, class_num]),
    'samme.r':
    voting_strategy.SAMME_R_voting_strategy,
    'jake_experimental':
    voting_strategy.JakeExperimentalVotingStrategy(class_num, args.blocks)
}

boosting_strategy_dict = {
    'naive': boosting_strategy.non_boosting_strategy,
    'samme.r': boosting_strategy.SAMME_R_boosting_strategy
}

# define the model
stem = stem_dict[args.stem](name='stem')
tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='stem')
blocks = [
    block_dict[args.block_type](name='block_{}'.format(i))
    for i in range(args.blocks)
]
classifiers = [
    classifier_dict[args.classifier](
        class_num=label_shape[1], name='classifier_{}'.format(i))
    for i in range(args.blocks)
]

classifier = boosted_classifier.BoostedClassifier(
    voting_strategy_dict[args.voting_strategy], blocks, classifiers,
    class_num)

# build model
data_ph = tf.placeholder(tf.float32, shape=data_shape)
label_ph = tf.placeholder(tf.int32, shape=label_shape)  # should be one-hot
stem_representation = stem(data_ph)
if args.load_stem:
    stem_representation = tf.stop_gradient(stem_representation)
final_classification, weak_logits = classifier(stem_representation)
weak_classifications = [tf.nn.softmax(logits) for logits in weak_logits]
wc_confusion_matrices = [
    tf.confusion_matrix(
        tf.argmax(label_ph, axis=1),
        tf.argmax(wl, axis=1),
        num_classes=class_num,
        dtype=tf.int32,
    ) for wl in weak_logits
]

# saver needs to be after building the model (so variables exist), but before
# the definiton of the optimizer
stem_saver = tf.train.Saver(
    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='stem'))

class_rate_fn = lambda a: tf.count_nonzero(
    tf.equal(tf.argmax(a, 1), tf.argmax(label_ph, 1)),
    dtype=tf.float32) / tf.constant(
        args.batch_size, dtype=tf.float32)
correct_weak_props = [class_rate_fn(wc) for wc in weak_classifications]
correct_final_prop = class_rate_fn(final_classification)


def feed_dict_fn():
    data, labels = next(train_gen)
    return {data_ph: data, label_ph: labels}


# calculate boosting weights
weights_list = boosting_strategy_dict[args.boosting_strategy](weak_logits,
                                                              label_ph)
losses_list = [
    tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.argmax(label_ph, axis=1), logits=wl) for wl in weak_logits
]
weighted_losses = [
    weights * raw_loss for weights, raw_loss in zip(weights_list, losses_list)
]

weights = tf.constant(
    1. / args.batch_size, dtype=tf.float32, shape=(args.batch_size, ))
weighted_losses = []
scale = -(float(class_num) - 1.) / float(class_num)

for i, classification in enumerate(weak_classifications):
    weighted_losses.append(
        weights * tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.argmax(label_ph, axis=1), logits=weak_logits[i]))
    weights = weights * tf.reduce_sum(
        tf.exp(scale *
               (tf.to_float(label_ph) * tf.log(classification + 1e-10))),
        axis=1)
    weight_means = tf.reduce_mean(weights)
    weights = weights / (weight_means * tf.to_float(args.batch_size))

# calculate gradients
optimizer = tf.train.AdamOptimizer()
grads_and_vars = [optimizer.compute_gradients(l) for l in weighted_losses]
vars = list(zip(*grads_and_vars[0]))[1]
grads = [[g if g is not None else tf.zeros_like(v) for g,v in gv]
         for gv in grads_and_vars]
total_grads = [sum(gs) for gs in list(zip(*grads))]
clipped_grads = [tf.clip_by_value(grad, -1., 1.) for grad in total_grads]
final_grads_and_vars = list(zip(clipped_grads, vars))
train_op = optimizer.apply_gradients(final_grads_and_vars)

pre_clip_wc_grads = [tf.global_norm(g) for g in grads]
pre_clip_global_grads = tf.global_norm(total_grads)
post_clip_global_grads = tf.global_norm(clipped_grads)

# if the voting strategy has an update fn, use it
# I, for one, welcome our new duck typing overlords
if hasattr(classifier.voting_strategy, 'update'):
    voting_strategy_update_op = classifier.voting_strategy.update(
        weak_logits, label_ph)
    train_op = tf.group(train_op, voting_strategy_update_op)

# if the voting strategy has an update fn, use it
# I, for one, welcome our new duck typing overlords
if hasattr(classifier.voting_strategy, 'update'):
    voting_strategy_update_op = classifier.voting_strategy.update(
        weak_logits, label_ph)
    train_op = tf.group(train_op, voting_strategy_update_op)

print("Trainable Parameters: {}".format(
    np.sum(
        [np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

verbose_ops_dict = dict()
verbose_ops_dict['weak_classifiers'] = weak_classifications
verbose_ops_dict['correct_weak_props'] = correct_weak_props
verbose_ops_dict['correct_final_prop'] = correct_final_prop
if args.gradient_norms:
    verbose_ops_dict['pre_clip_global_grads'] = pre_clip_global_grads
    verbose_ops_dict['post_clip_global_grads'] = post_clip_global_grads
    verbose_ops_dict['pre_clip_wc_grads'] = pre_clip_wc_grads
if args.weak_classifier_conf_matrix:
    verbose_ops_dict['wc_conf_matrices'] = wc_confusion_matrices

# initialize session and train
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as session:
    session.run(tf.global_variables_initializer())
    if args.load_stem:
        stem_saver.restore(session, args.stem)
    for epoch in range(args.epochs):
        print("EPOCH {}".format(epoch))
        outs = util.run_epoch_ops(
            session,
            train_data.shape[0] // args.batch_size,
            silent_ops=[train_op],
            verbose_ops=verbose_ops_dict,
            feed_dict_fn=feed_dict_fn,
            verbose=True)

        # Process metrics
        final_acc_val = outs['correct_final_prop']
        mean_final_acc_val = np.mean(final_acc_val)
        print("Final Accuracy: " + str(mean_final_acc_val))

        weak_acc_vals = outs['correct_weak_props']
        mean_weak_acc_vals = np.mean(weak_acc_vals, axis=0)
        print("Weak Classifier Accuracies: " + str(mean_weak_acc_vals))

        if args.gradient_norms:
            pre_clip_global_grad_vals = outs['pre_clip_global_grads']
            mean_pre_clip_global_grad_vals = np.mean(pre_clip_global_grad_vals)
            post_clip_global_grad_vals = outs['post_clip_global_grads']
            mean_post_clip_global_grad_vals = np.mean(
                post_clip_global_grad_vals)
            pre_clip_wc_grad_vals = outs['pre_clip_wc_grads']
            mean_pre_clip_wc_grad_vals = np.mean(pre_clip_wc_grad_vals, axis=0)
            with open(os.path.join(args.log_dir, 'grad_norms.csv'),
                      'a') as csv:
                csv.write('{},{},{}\n'.format(
                    mean_pre_clip_global_grad_vals,
                    mean_post_clip_global_grad_vals, ','.join([
                        '{}'.format(val) for val in mean_pre_clip_wc_grad_vals
                    ])))

        if args.weak_classifier_covar:
            weak_classification_vals = np.reshape(
                np.swapaxes(outs['weak_classifiers'], 1, 2), (-1, 5, 10))
            mean_weak_classification_covar_vals = np.mean(
                np.array([
                    np.cov(x, rowvar=False) for x in weak_classification_vals
                ]),
                axis=0)
            with open(
                    os.path.join(args.log_dir, 'weak_learner_covariance.csv'),
                    'a') as csv:
                csv.write('{}\n'.format(mean_weak_classification_covar_vals))

        # plots
        if args.weak_classifier_rate:
            util.plot_parallel_histograms(
                values=np.swapaxes(weak_acc_vals, 0, 1),
                save_path=os.path.join(
                    args.log_dir,
                    'weak_learner_classification_rate_epoch_{}'.format(epoch)),
                hist_title='Weak Learner',
                xlabel='Correct Rate',
                ylabel='Minibatches')

        if args.weak_classifier_conf_matrix:
            wc_conf_matrix_vals = np.sum(outs['wc_conf_matrices'], axis=0)
            [
                util.plot_confusion_matrix(
                    cm,
                    CIFAR10_CLASS_NAMES,
                    title='confusion_matrix_epoch_{}_wc_{}'.format(epoch, i),
                    log_dir=os.path.join(args.log_dir,
                                         'wc_confusion_matrices'))
                for i, cm in enumerate(wc_conf_matrix_vals)
            ]
