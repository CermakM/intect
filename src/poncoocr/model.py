"""Module containing model of convolutional neural network for the poncoocr engine."""

import numpy as np
import tensorflow as tf

import names

from . import architecture, utils
from . import layers

__activation_dict = utils.AttrDict(
    relu=tf.nn.relu, sigmoid=tf.nn.sigmoid, tanh=tf.nn.tanh
)

__layer_dict = utils.AttrDict(
    conv=layers.ConvLayer, fcl=layers.FullyConnectedLayer
)


class CNN:

    __model_names = set()

    def __init__(self, name: str = None):
        """Initialize the model."""
        if name is None:
            # Generate some random name
            rand_name = names.get_first_name(gender='female')
            while rand_name in self.__model_names:
                rand_name = names.get_first_name(gender='female')
            self._name = tf.Variable(rand_name, dtype=tf.string)
            self.__model_names.add(self._name)

        # initialize the tf session
        tf.reset_default_graph()
        self._session = tf.Session()

        self._batch_size = tf.placeholder(tf.uint8, name='batch_size')
        self._learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self._padding = tf.placeholder(tf.string, name='padding')
        self._stride = tf.placeholder(tf.uint8, name='stride')

    @property
    def name(self):
        return self._name

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def padding(self):
        return self._padding

    @property
    def stride(self):
        return self._stride

    @classmethod
    def from_architecture(cls, arch):
        pass

    def train(self):
        pass

    def predict(self):
        pass

    def save(self):
        pass

    def add_layer(self, layer):
        """Add layer to the model."""
        pass
