import sonnet as snt
import tensorflow as tf


def sequential_conv_block():
    return snt.Sequential([
        snt.Residual(snt.Conv2D(32, 3)), tf.nn.elu,
        snt.Residual(snt.Conv2D(32, 3)), tf.nn.elu,
        snt.Residual(snt.Conv2D(32, 3)), tf.nn.elu
    ])
