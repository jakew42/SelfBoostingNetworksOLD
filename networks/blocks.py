import sonnet as snt
import tensorflow as tf


class ResidualConvBlock(snt.AbstractModule):
    def __init__(self, channels=32, name='residual_conv_block'):
        super(ResidualConvBlock, self).__init__(name=name)
        with self._enter_variable_scope():
            self._block = snt.Sequential([
                snt.Residual(snt.Conv2D(channels, 3)), tf.nn.elu,
                snt.Residual(snt.Conv2D(channels, 3)), tf.nn.elu,
                snt.Residual(snt.Conv2D(channels, 3)), tf.nn.elu
            ])

    def _build(self, inputs):
        return self._block(inputs)

class IdentityBlock(snt.AbstractModule):
    def __init__(self, name='identity_block'):
        super(IdentityBlock, self).__init__(name=name)
        with self._enter_variable_scope():
            self._block = tf.identity

    def _build(self, inputs):
        return self._block(inputs)

