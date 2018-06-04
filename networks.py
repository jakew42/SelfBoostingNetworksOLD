import sonnet as snt
import tensorflow as tf


class ResidualConvBlock(snt.AbstractModule):
    def __init__(self, channels, name='residual_conv_block'):
        super(ResidualConvBlock, self).__init__(name=name)
        with self._enter_variable_scope():
            self._block = snt.Sequential([
                snt.Residual(snt.Conv2D(channels, 3)), tf.nn.elu,
                snt.Residual(snt.Conv2D(channels, 3)), tf.nn.elu,
                snt.Residual(snt.Conv2D(channels, 3)), tf.nn.elu
            ])

    def _build(self, inputs):
        return self._block(inputs)

class ResidualClassifier(snt.AbstractModule):
    def __init__(self, num_blocks, class_num, name='residual_classifier'):
        super(ResidualClassifier, self).__init__(name=name)
        with self._enter_variable_scope():
            entry_layer = snt.Sequential(
                [snt.Conv2D(32, 3, name='entry_conv2d'), tf.nn.elu])
            blocks = snt.Sequential([ResidualConvBlock(32) for _ in range(num_blocks)])
            classifier = snt.Linear(class_num)
            self._net = snt.Sequential([entry_layer, blocks, tf.layers.Flatten(), classifier])

    def _build(self, inputs):
        return self._net(inputs)
