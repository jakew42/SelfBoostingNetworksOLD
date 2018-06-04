import tensorflow as tf
import sonnet as snt

import networks


class NaiveBoostedClassifier(snt.AbstractModule):
    def __init__(self, num_blocks, class_num, name='boosted_classifier'):
        super(NaiveBoostedClassifier, self).__init__(name=name)
        self._class_num = class_num

        self._blocks = []
        self._classifiers = []
        with self._enter_variable_scope():
            self._entry_layer = snt.Sequential(
                [snt.Conv2D(32, 3, name='entry_conv2d'), tf.nn.elu])
            for i in range(num_blocks):
                self._blocks.append(snt.Sequential([
                snt.Residual(snt.Conv2D(32, 3)), tf.nn.elu,
                snt.Residual(snt.Conv2D(32, 3)), tf.nn.elu,
                snt.Residual(snt.Conv2D(32, 3)), tf.nn.elu
            ]))
                self._classifiers.append(
                    snt.Sequential([
                        snt.Conv2D(3, 3, name='classifier_conv2d_{}'.format(i)),
                        tf.layers.Flatten(),
                        snt.Linear(class_num, name='classifier_linear_{}'.format(i))
                    ]))

    def _build(self, inputs):
        logits = []
        h_ks = []
        x = self._entry_layer(inputs)
        for i, _ in enumerate(self._blocks):
            x = self._blocks[i](x)
            c = self._classifiers[i](x)
            logits.append(c)
            probs = tf.nn.softmax(c)
            hk_inner_prod = tf.constant(
                (1 / self._class_num),
                dtype=tf.float32,
                shape=(self._class_num, self._class_num))
            hk_inner_prod = tf.matrix_set_diag(hk_inner_prod,
                                               tf.ones([self._class_num]))
            block_h_k = (1 / self._class_num) * tf.matmul(probs, hk_inner_prod)
            h_ks.append(block_h_k)

        final_classification = tf.accumulate_n(h_ks)
        return final_classification, logits
