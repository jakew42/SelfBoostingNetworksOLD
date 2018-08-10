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

def selection_decorator(func, idxs):
    def func_wrapper(logits, labels):
        weights_list = func(logits, labels)
        scale = [0.] * len(weights_list)
        for idx in idxs:
            scale[idx] = 1.
        return [w * s for w, s in zip(weights_list, scale)]

# calculate boosting weights
def calculate_boosted_losses(boosting_strategy_fn, weak_logits, label_ph, batch_size, class_num):
    weights_list = boosting_strategy_fn(weak_logits, label_ph)
    losses_list = [
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.argmax(label_ph, axis=1), logits=wl) for wl in weak_logits
    ]
    weighted_losses = [
        weights * raw_loss for weights, raw_loss in zip(weights_list, losses_list)
    ]

    weights = tf.constant(
        1. / batch_size, dtype=tf.float32, shape=(batch_size, ))
    weighted_losses = []
    scale = -(float(class_num) - 1.) / float(class_num)

    weak_classifications = [tf.nn.softmax(logits) for logits in weak_logits]
    for i, classification in enumerate(weak_classifications):
        weighted_losses.append(
            tf.stop_gradient(weights) * tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.argmax(label_ph, axis=1), logits=weak_logits[i]))
        weights = weights * tf.reduce_sum(
            tf.exp(scale *
                (tf.to_float(label_ph) * tf.log(classification + 1e-10))),
            axis=1)
        weight_means = tf.reduce_mean(weights)
        weights = weights / (weight_means * tf.to_float(batch_size))
    return weighted_losses

def calculate_boosted_gradients(optimizer, weighted_losses, weights_scale):
    grads_and_vars = [optimizer.compute_gradients(l) for l in weighted_losses]
    vars = list(zip(*grads_and_vars[0]))[1]
    grads = [[g if g is not None else tf.zeros_like(v) for g, v in gv]
            for gv in grads_and_vars]
    grads = [[g * s for g in gs]
            for gs, s in zip(grads, tf.split(weights_scale, len(grads)))]
    total_grads = [sum(gs) for gs in list(zip(*grads))]
    clipped_grads = [tf.clip_by_value(grad, -1., 1.) for grad in total_grads]
    final_grads_and_vars = list(zip(clipped_grads, vars))

    metrics = dict()
    metrics['pre_clip_wc_grads'] = [tf.global_norm(g) for g in grads]
    metrics['pre_clip_global_grads'] = tf.global_norm(total_grads)
    metrics['post_clip_global_grads'] = tf.global_norm(clipped_grads)

    return final_grads_and_vars, metrics
