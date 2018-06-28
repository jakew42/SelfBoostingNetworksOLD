import tensorflow as tf
import sonnet as snt

import networks


class BoostedClassifier(snt.AbstractModule):
    """
    Classifier module which performs self-boosting using a provided network,
    voting strategy, and boosting strategy. (TODO: the latter two)
    """

    def __init__(self,
                 stem,
                 blocks,
                 classifiers,
                 class_num,
                 name='boosted_classifier'):
        """
        Args:
           stem: An initial module to preprocess the input
           blocks: A list of modules, applied in succession after stem
           classifiers: A list parallel to blocks, to be weak learners
        """
        super(BoostedClassifier, self).__init__(name=name)
        assert len(blocks) == len(
            classifiers), 'Must have equal number of blocks and classifiers'
        self._blocks = blocks
        self._classifiers = classifiers
        self._stem = stem
        self._class_num = class_num

    def _build(self, inputs):
        logits = []
        h_ks = []
        x = self._stem(inputs)
        for i, _ in enumerate(self._blocks):
            x = self._blocks[i](x)
            c = self._classifiers[i](x)
            logits.append(c)
            probs = tf.log(tf.nn.softmax(c))
            hk_inner_prod = tf.constant(
                (-1 / self._class_num),
                dtype=tf.float32,
                shape=(self._class_num, self._class_num))
            hk_inner_prod = tf.matrix_set_diag(hk_inner_prod,
                                               tf.ones([self._class_num]))
            block_h_k = (self._class_num - 1) * tf.matmul(probs, hk_inner_prod)
            h_ks.append(block_h_k)

        final_classification = tf.accumulate_n(h_ks)
        return final_classification, logits
