"""Module implementing dnn layers."""

from abc import abstractmethod

import tensorflow as tf
import typing


class Layer(object):
    """Layer base class."""

    def __init__(self,
                 input_data,
                 input_channels: int,
                 output_channels: int,
                 activation: typing.Callable,
                 name: str
                 ):
        self.input_data = input_data
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.activation = activation
        self.name = name

    @abstractmethod
    def activate(self):
        raise NotImplementedError


class ConvLayer(Layer):
    """Convolution layer class."""

    def __init__(self,
                 input_data,
                 input_channels: int,
                 output_channels: int,
                 activation: typing.Callable,
                 name: str,
                 filter_shape: tuple = (3, 3),
                 stride: tuple = (1, 1, 1, 1),
                 padding: str = 'SAME',
                 ):
        """Initialize Convolution layer."""
        # TODO: check if the filter shape and stride is correct
        self.filter_shape = filter_shape
        self.stride = stride
        self.padding = padding

        super().__init__(input_data, input_channels, output_channels, activation, name)

    def activate(self):
        with tf.name_scope(self.name):
            # initialize weights
            w_init = tf.truncated_normal(
                shape=(*self.filter_shape, self.input_channels, self.output_channels),
                stddev=0.1
            )
            w = tf.Variable(initial_value=w_init, name='W')
            b = tf.Variable(initial_value=tf.constant(0.1, shape=[self.output_channels], name='B'))
            conv = tf.nn.conv2d(input=self.input_data,
                                filter=w,
                                padding=self.padding,
                                strides=self.stride,
                                )

            # apply activation
            activated = self.activation(conv + b)

            # Add summaries
            tf.summary.histogram(values=w, name='weights')
            tf.summary.histogram(values=b, name='biases')
            tf.summary.histogram(values=activated, name='activations')

            # Use default max pooling here for now, maybe parametrize this later
            return tf.nn.max_pool(activated, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                                  padding="SAME")


class FullyConnectedLayer(Layer):

    def __init__(self,
                 input_data,
                 input_channels: int,
                 output_channels: int,
                 activation: typing.Callable,
                 name: str,
                 ):
        """Initialize fully connected layer."""
        super().__init__(input_data, input_channels, output_channels, activation, name)

    def activate(self):
        # initialize wights
        w_init = tf.truncated_normal(
            shape=(self.input_channels, self.output_channels),
            stddev=0.1
        )
        w = tf.Variable(initial_value=w_init, name='W')
        b = tf.Variable(initial_value=tf.constant(0.1, shape=[self.output_channels]), name='b')

        # apply activation
        activated = self.activation(tf.matmul(self.input_data, w) + b)

        # add summaries
        tf.summary.histogram(values=w, name='weights')
        tf.summary.histogram(values=b, name='biases')
        tf.summary.histogram(values=activated, name='activations')

        return activated
