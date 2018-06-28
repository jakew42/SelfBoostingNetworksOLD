import sonnet as snt
import tensorflow as tf


class ReduceFlattenClassifier(snt.AbstractModule):
    def __init__(self, class_num, name='reduce_flatten_classifier'):
        super(ReduceFlattenClassifier, self).__init__(name=name)
        with self._enter_variable_scope():
            self._block = snt.Sequential([
                            snt.Conv2D(3, 3),
                            tf.layers.Flatten(),
                            snt.Linear(class_num)
            ])
    def _build(self, inputs):
        return self._block(inputs)
