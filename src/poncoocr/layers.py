"""Module implementing dnn layers."""

from abc import abstractmethod

import tensorflow as tf
import typing

from . import utils


class Layer(object):
    """Layer base class."""

    def __init__(self,
                 output_channels: int,
                 activation: typing.Callable,
                 input_data=None):
        """Initialize the layer class.
        :param input_data: np.array or tf.Tensor, object must implement `shape` function
        :param output_channels: int, number of channels to be used in the current layer (ie. for
        convolutional layers is this number the number of filters used for convolution.
        :param activation: callable, activation function to be used on the resulting tensor
        :param name: name of the layer currently one of {conv, fcl}
        """
        self.input_data = input_data
        self.input_shape = input_data.shape if input_data is not None else None
        self.output_channels = output_channels
        self.activation = activation

    @property
    def shape(self):
        return self.input_shape, self.output_channels

    def connect(self, input_layer: typing.Union["Layer", tf.Tensor, typing.Sequence]):
        """Connects the current layer to the `input_layer`."""
        self.input_data = input_layer
        self.input_shape = input_layer.shape

        return self

    @abstractmethod
    def activate(self):
        raise NotImplementedError


class ConvLayer(Layer):
    """Convolution layer class."""

    def __init__(self,
                 output_channels: int,
                 activation: typing.Callable = None,
                 filter_shape: tuple = (3, 3),
                 stride: tuple = (1, 1, 1, 1),
                 padding: str = 'SAME',
                 input_data=None,
                 **kwargs):
        """Initialize Convolution layer."""
        self.name = 'conv'
        activation = activation or tf.train.AdamOptimizer

        self.filter_shape = filter_shape
        self.stride = stride
        self.padding = padding
        self.params = utils.AttrDict(**kwargs)

        super().__init__(input_data=input_data,
                         output_channels=output_channels,
                         activation=activation,
                         )

    def activate(self):
        """Activates the layer."""
        if not all([self.input_data, self.input_shape]):
            raise AttributeError("`input_data` has not been provided."
                                 "Use `connect` method to connect the layer to another.")

        with tf.name_scope(self.name):
            # initialize weights
            w_init = tf.truncated_normal(
                # Use the input shape but change the number of channels
                shape=(self.input_shape[:-1], self.output_channels),
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
                 output_channels: int,
                 activation: typing.Callable = None,
                 input_data=None,
                 **kwargs):
        """Initialize fully connected layer."""
        self.name = 'fcl'
        activation = activation or tf.train.AdamOptimizer
        self.params = utils.AttrDict(**kwargs)
        super().__init__(input_data=input_data,
                         output_channels=output_channels,
                         activation=activation,
                         )

    def activate(self):
        # initialize wights
        with tf.name_scope(self.name):
            w_init = tf.truncated_normal(
                shape=(self.input_shape[:-1], self.output_channels),
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
