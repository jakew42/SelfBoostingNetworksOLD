"""
A boosting strategy is a callable that takes logits from weak learners and correct labels and 
returns list of boosting weightings (per weak classifier). This project uses those weights to scale gradients to
train different gradient-based weak learners.
"""
import tensorflow as tf
import sonnet as snt


def boosting_strategy(logits, labels):
    pass


def non_boosting_strategy(logits, labels):
    block_num = len(logits)
    batch_size = labels.get_shape().as_list()[0]
    return [tf.ones([batch_size]) / float(batch_size)] * block_num


def SAMME_R_boosting_strategy(logits, labels):
    """labels should be one-hot"""
    batch_size = labels.get_shape().as_list()[0]
    class_num = labels.get_shape().as_list()[-1]

    scale = -(float(class_num) - 1.) / float(class_num)
    weights = tf.constant(
        1. / batch_size, dtype=tf.float32, shape=(batch_size, ))
    weights_list = [weights]

    for logit in logits:
        log_likelihood = tf.log(tf.nn.softmax(logit) + 1e-10)
        weights *= tf.reduce_sum(
            tf.exp(scale * tf.to_float(labels) * log_likelihood), axis=1)
        weight_means = tf.reduce_mean(weights)
        weights = weights / (weight_means * tf.to_float(batch_size))
        weights_list.append(weights)

    return weights_list
