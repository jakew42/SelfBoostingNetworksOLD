import tensorflow as tf
import sonnet as snt


class NaiveBoostedClassifier(snt.AbstractModule):
    def __init__(self, num_blocks, class_num, name='boosted_classifier'):
        super(NaiveBoostedClassifier, self).__init__(name=name)

        self._blocks = []
        self._classifiers = []
        self._alphas = []
        with self._enter_variable_scope():
            self._entry_layer = snt.Conv2D(32, 3)
            for i in range(num_blocks):
                self._blocks.append(
                    snt.Sequential([
                        snt.Residual(snt.Conv2D(32, 3)),
                        snt.Residual(snt.Conv2D(32, 3)),
                        snt.Residual(snt.Conv2D(32, 3))
                    ]))
                self._classifiers.append(
                    snt.Sequential(
                        [tf.layers.Flatten(),
                         snt.Linear(class_num)]))
                self._alphas.append(
                    tf.get_variable(
                        'alpha_{}'.format(i),
                        initializer=tf.constant(1.),
                        dtype=tf.float32,
                        trainable=False),
                )  # THIS SHOULD BE UPDATED TO ADABOOST NUM

    def _build(self, inputs):
        logits = []
        contributions = []
        x = self._entry_layer(inputs)
        for i, _ in enumerate(self._blocks):
            x = self._blocks[i](x)
            c = self._classifiers[i](x)
            logits.append(c)
            contributions.append(self._alphas[i] * c)

    #p_k = tf.reduce_sum(label_ph * classification, axis=1)
    #p_other = (tf.reduce_sum(classification, axis=1) - p_k) / class_num
    #h_k = (class_num - 1)(tf.log(p_k) - p_other)

        final_classification = tf.accumulate_n(
            contributions) / tf.accumulate_n(self._alphas)
        return final_classification, logits
