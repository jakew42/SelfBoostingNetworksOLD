import tensorflow as tf
import sonnet as snt

import networks


class BoostedClassifier(snt.AbstractModule):
    """
    Classifier module which performs self-boosting using a provided network,
    voting strategy, and boosting strategy. (TODO: the latter two)
    """

    def __init__(self,
                 voting_strategy,
                 blocks,
                 classifiers,
                 class_num,
                 name='boosted_classifier'):
        """
        Args:
          voting_strategy: A callable which takes a list of logits and returns
                           the final, boosted classification
          blocks: A list of modules, applied in succession after stem
          classifiers: A list parallel to blocks, to be weak learners
        """
        super(BoostedClassifier, self).__init__(name=name)
        self.voting_strategy = voting_strategy
        assert len(blocks) == len(
            classifiers), 'Must have equal number of blocks and classifiers'
        self._blocks = blocks
        self._classifiers = classifiers
        self._class_num = class_num

    def _build(self, inputs):
        x = inputs
        logits = []
        for i, _ in enumerate(self._blocks):
            x = self._blocks[i](x)
            c = self._classifiers[i](x)
            logits.append(c)

        final_classification = self.voting_strategy(logits)
        return final_classification, logits


def build_model(stem_fn,
                block_fn,
                classifier_fn,
                block_num,
                voting_strategy_fn,
                batch_size,
                class_num,
                data_shape,
                label_shape,
                load_stem=False):
    stem = stem_fn(name='stem')
    blocks = [block_fn(name='block_{}'.format(i)) for i in range(block_num)]
    classifiers = [
        classifier_fn(class_num=class_num, name='classifier_{}'.format(i))
        for i in range(block_num)
    ]
    classifier = BoostedClassifier(voting_strategy_fn, blocks, classifiers,
                                   class_num)

    # build model
    data_ph = tf.placeholder(tf.float32, shape=(batch_size, ) + data_shape[1:])
    label_ph = tf.placeholder(
        tf.int32, shape=(batch_size, ) + label_shape[1:])  # should be one-hot
    stem_representation = stem(data_ph)
    if load_stem:
        stem_representation = tf.stop_gradient(stem_representation)
    final_classification, weak_logits = classifier(stem_representation)
    weak_classifications = [tf.nn.softmax(logits) for logits in weak_logits]

    metrics = dict()
    metrics['weak_classifications'] = weak_classifications
    metrics['wc_confusion_matrices'] = [
        tf.confusion_matrix(
            tf.argmax(label_ph, axis=1),
            tf.argmax(wl, axis=1),
            num_classes=class_num,
            dtype=tf.int32,
        ) for wl in weak_logits
    ]
    class_rate_fn = lambda a: tf.count_nonzero(
        tf.equal(tf.argmax(a, 1), tf.argmax(label_ph, 1)),
        dtype=tf.float32) / tf.constant(
            batch_size, dtype=tf.float32)
    metrics['correct_weak_props'] = [
        class_rate_fn(wc) for wc in weak_classifications
    ]
    metrics['correct_final_prop'] = class_rate_fn(final_classification)
    metrics[
        'final_classification_loss'] = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=final_classification, labels=tf.argmax(label_ph, axis=1))

    return data_ph, label_ph, final_classification, weak_logits, classifier, metrics
