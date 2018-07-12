"""
A voting strategy is a callable that takes logits from weak learners and combines them into a final
classification.

If the strategy has parameters, it may inherit from `snt.AbstractModule` and optionally implement
an update method which takes (logits, labels) and returns a TF operation.
"""
import tensorflow as tf
import sonnet as snt


def voting_strategy(logits):
    """
    Abstract base function for accumulating the votes of weak learners into an
    overall final classification.

    Args:
      logits: A list of tensors with logits from each weak learner.
              Should be non-empty and each member should have identical shape.
              The number of classes is inferred from length of the last dim.

    Returns:
      A tensor 
    """
    class_num = logits.get_shape().as_list()[-1]
    for x in logits:
        assert x.shape == logits[0].shape
    pass


def naive_voting_strategy(logits):
    for x in logits:
        assert x.shape == logits[0].shape
    return tf.accumulate_n(logits) / float(len(logits))


class LinearComboStrategy(snt.AbstractModule):
    def __init__(self, weak_learner_num, name='linear_combo_strategy'):
        super(LinearComboStrategy, self).__init__(name=name)
        with self._enter_variable_scope():
            self._weights = tf.get_variable(
                name + '_weights',
                shape=(weak_learner_num),
                initializer=tf.ones_initializer,
                dtype=tf.float32,
                trainable=False)
            self.optimizer = tf.train.AdamOptimizer()

    def _build(self, logits):
        assert len(logits) == self._weights.get_shape().as_list()[0]
        stopped_logits = [tf.stop_gradient(l) for l in logits]
        weighted_logits = [
            a * b for a, b in zip(stopped_logits,
                                  tf.split(self._weights, len(logits)))
        ]
        return tf.accumulate_n(weighted_logits) / float(len(logits))

    def update(self, logits, labels):
        final_logits = self._build(logits)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.argmax(labels, axis=1), logits=final_logits)
        return self.optimizer.minimize(loss, var_list=[self._weights])


class MLPComboStrategy(snt.AbstractModule):
    def __init__(self, output_sizes, name='mlp_combo_strategy'):
        super(MLPComboStrategy, self).__init__(name=name)
        with self._enter_variable_scope():
            self._model = snt.Sequential([
                tf.layers.Dense(size, activation=tf.nn.relu, trainable=False)
                for size in output_sizes
            ])
            self.optimizer = tf.train.AdamOptimizer()

    def _build(self, logits):
        print('built!')
        return self._model(tf.concat(logits, axis=1))

    def update(self, logits, labels):
        final_logits = self._build(logits)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.argmax(labels, axis=1), logits=final_logits)
        train_vars = [l.non_trainable_variables for l in self._model.layers]
        print(train_vars)
        train_vars = [item for sublist in train_vars for item in sublist]
        print(train_vars)
        return self.optimizer.minimize(loss, var_list=train_vars)

def SAMME_R_voting_strategy(logits):
    """
    Algorithm 4 of "Multi-class AdaBoost" by Zhu et al. 2006

    PDF: Can be found at the bottom of page 9
    (https://web.stanford.edu/~hastie/Papers/samme.pdf)

    Args:
      See `voting strategy`
    """
    class_num = logits[0].get_shape().as_list()[-1]
    for x in logits:
        assert x.shape == logits[0].shape

    log_probs = [tf.log(tf.nn.softmax(l)) for l in logits]
    # two steps to get a matrix of -1 except for the diagonal which is 1
    hk_inner_prod = tf.constant(
        (-1 / class_num), dtype=tf.float32, shape=(class_num, class_num))
    hk_inner_prod = tf.matrix_set_diag(hk_inner_prod, tf.ones([class_num]))
    h_ks = [(class_num - 1) * tf.matmul(lp, hk_inner_prod) for lp in log_probs]

    return tf.accumulate_n(h_ks)


class JakeExperimentalVotingStrategy(snt.AbstractModule):
    def __init__(self, class_num, weak_learner_num, name='jakes_loco_strat'):
        super(JakeExperimentalVotingStrategy, self).__init__(name=name)
        with self._enter_variable_scope():
            self._weak_running_sums = [
                tf.get_variable(
                    'wrs_{}'.format(i),
                    initializer=tf.ones(class_num),
                    trainable=False) for i in range(weak_learner_num)
            ]

            self._running_accs = [
                tf.get_variable(
                    'acc_{}'.format(i),
                    initializer=tf.ones(1),
                    trainable=False) for i in range(weak_learner_num)
            ]

    def _build(self, logits):
        weak_classifications = [tf.nn.softmax(logits) for logits in logits]
        weighted_classifications = [
            c * (1. / (s + 1e-5) * a)
            for c, s, a in zip(weak_classifications, self._weak_running_sums,
                               self._running_accs)
        ]
        return tf.accumulate_n(weighted_classifications)

    def update(self, logits, labels):
        batch_size = labels.get_shape().as_list()[0]
        weak_classifications = [tf.nn.softmax(logits) for logits in logits]
        weak_sums = [
            tf.square(wc - tf.to_float(labels)) for wc in weak_classifications
        ]
        weak_sum_totals = [tf.reduce_mean(ws, 0) for ws in weak_sums]

        class_rate_fn = lambda a: tf.count_nonzero(
            tf.equal(tf.argmax(a, 1), tf.argmax(labels, 1)),
            dtype=tf.float32) / tf.constant(batch_size, dtype=tf.float32)
        correct_weak_props = [class_rate_fn(wc) for wc in weak_classifications]

        weak_running_sums_updates = [
            tf.assign(wrs, .95 * wrs + .05 * ws)
            for wrs, ws in zip(self._weak_running_sums, weak_sum_totals)
        ]
        running_accs_updates = [
            tf.assign(wrs, .95 * wrs + .05 * ws)
            for wrs, ws in zip(self._running_accs, correct_weak_props)
        ]

        return tf.group([weak_running_sums_updates, running_accs_updates])
