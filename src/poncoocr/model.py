"""Module containing model of convolutional neural network for the poncoocr engine."""

import sys
import tempfile
import typing

import names
import tensorflow as tf

from . import architecture


class Model(object):

    __model_names = set()

    def __init__(self,
                 inputs,
                 labels,
                 name: str = None,
                 params: dict = None):
        """Initialize the model."""
        if name is None:
            # Generate some random name
            rand_name = names.get_first_name(gender='female')
            while rand_name in self.__model_names:
                rand_name = names.get_first_name(gender='female')
            self._name = rand_name
            self.__model_names.add(self._name)

        self._x = tf.placeholder(tf.float32, shape=inputs.shape)
        self._labels = tf.placeholder(tf.float32, shape=labels.shape)

        self._layers = [self._x]

        if params is None:
            params = dict()

        # Configurable and directly accessible properties
        self.batch_size = tf.constant(getattr(params, 'batch_size', 32), tf.uint8)
        self.learning_rate = tf.constant(getattr(params, 'learning_rate', 1E-4), tf.float32)
        self.optimizer = getattr(tf.train, getattr(params, 'optimizer', 'AdamOptimizer'), tf.train.AdamOptimizer)

        # Directory to save the model to
        self.model_dir = getattr(params, 'model_dir', tempfile.mkdtemp(prefix=self._name))

    def __repr__(self):
        return "<class 'poncoocr.model.Model'" \
               "  name: {s._name}" \
               "  layers: {s._layers}>".format(s=self)

    @property
    def x(self):
        return self._x

    @property
    def labels(self):
        return self._labels

    @property
    def logits(self):
        return self._layers[-1]

    @property
    def name(self):
        return self._name

    @property
    def input_layer(self):
        return self._layers[0]

    @property
    def hidden_layers(self):
        return tuple(self._layers[1:])

    @classmethod
    def from_architecture(cls, inputs, labels, arch: architecture.ModelArchitecture, params: dict = None):
        """Initialize the model from the given Architecture"""
        # load network parameters
        if arch.name == 'default':
            name = None
        else:
            name = arch.name

        arch_params = dict(
            # model parameters
            batch_size=arch.batch_size,
            learning_rate=arch.learning_rate,
            optimizer=getattr(tf.train, arch.optimizer, tf.train.AdamOptimizer),
        )

        if params:
            arch_params.update(params)

        model = cls(
            inputs=inputs,
            labels=labels,
            name=name,
            params=params
        )

        # Load each layer from architecture and add it to the model
        for layer in arch.layers:
            # Construct a layer by the type specified in the architecture
            config = layer.params

            if layer.type == 'conv2d':
                model.add_conv_layer(**config)

            elif layer.type == 'max_pooling2d':
                model.add_max_pooling_layer(**config)

            elif layer.type == 'flatten':
                model.add_flatten_layer()

            elif layer.type == 'dense':
                model.add_dense_layer(**config)

            else:
                print("Invalid type of layer provided: `%s`. Skipping." % layer.type, file=sys.stderr)

        return model

    def add_conv_layer(self,
                       filters: int,
                       kernel_size: typing.Union[typing.Sequence, tf.TensorShape],
                       activation: typing.Union[typing.Callable, str] = None,
                       strides: typing.Tuple[int, int] = (1, 1),
                       padding='same',
                       name=None,
                       *args, **kwargs):

        # If activation is provided as a string, i.e `relu`, get the corresponding activation function
        if isinstance(activation, str):
            activation = getattr(tf.nn, activation, None)

        # Uniquify layer name
        layer_name = "{name}_{id}".format(name=name or getattr(kwargs, 'type', 'conv'), id=len(self._layers))

        with tf.name_scope(self._name):
            # initialize weights
            conv = tf.layers.conv2d(
                inputs=self._layers[-1],
                filters=filters,
                activation=activation,
                kernel_size=kernel_size,
                padding=padding,
                strides=strides,
                name=layer_name,
                *args, **kwargs
            )

            # Add summaries
            # TODO

        self._layers.append(conv)

    def add_flatten_layer(self):
        self._layers.append(tf.layers.flatten(inputs=self._layers[-1]))

    def add_dense_layer(self,
                        units: int,
                        activation: typing.Union[typing.Callable, str] = None,
                        name=None,
                        *args, **kwargs):

        # If activation is provided as a string, i.e `relu`, get the corresponding activation function
        if isinstance(activation, str):
            activation = getattr(tf.nn, activation, None)

        # Uniquify layer name
        layer_name = "{name}_{id}".format(name=name or getattr(kwargs, 'type', 'dense'), id=len(self._layers))

        with tf.name_scope(self._name):
            dense = tf.layers.dense(
                inputs=self._layers[-1],
                units=units,
                activation=activation,
                name=layer_name,
                *args, **kwargs
            )

            # add summaries
            # TODO

        self._layers.append(dense)

    def add_max_pooling_layer(self,
                              pool_size: typing.Union[typing.Sequence, tf.TensorShape],
                              strides: int = 2,
                              name=None,
                              *args, **kwargs):

        # Uniquify layer name
        layer_name = "{name}_{id}".format(name=name or getattr(kwargs, 'type', 'pool'), id=len(self._layers))

        with tf.name_scope(self._name):
            pool = tf.layers.max_pooling2d(
                inputs=self._layers[-1],
                pool_size=pool_size,
                strides=strides,
                name=layer_name,
                *args, **kwargs
            )

            self._layers.append(pool)

    def save(self):
        raise NotImplementedError
