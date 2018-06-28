import sonnet as snt
import tensorflow as tf



class BigConvStem(snt.AbstractModule):
    def __init__(self, channels=32, name='big_conv_stem'):
        super(BigConvStem, self).__init__(name=name)
        with self._enter_variable_scope():
            self._block = snt.Sequential([
                snt.Conv2D(channels, 3), tf.nn.elu] + 
                [snt.Residual(snt.Conv2D(channels, 3)), tf.nn.elu]*10
            )

    def _build(self, inputs):
        return self._block(inputs)
