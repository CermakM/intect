"""Module containing model of convolutional neural network for the poncoocr engine."""

import typing
import tensorflow as tf

import names

from . import architecture, utils
from . import layers

_activation_dict = dict(
   relu=tf.nn.relu, sigmoid=tf.nn.sigmoid, tanh=tf.nn.tanh
)

_layer_dict = dict(
    conv=layers.ConvLayer, fcl=layers.FullyConnectedLayer
)


class CNN(object):

    __model_names = set()

    def __init__(self,
                 input_shape,
                 output_shape,
                 hidden_layers: list = None,
                 name: str = None,
                 params: dict = None):
        """Initialize the model."""
        # initialize the tf session
        tf.reset_default_graph()
        self._session = tf.Session()

        if name is None:
            # Generate some random name
            rand_name = names.get_first_name(gender='female')
            while rand_name in self.__model_names:
                rand_name = names.get_first_name(gender='female')
            self._name = tf.Variable(rand_name, dtype=tf.string)
            self.__model_names.add(self._name)

        self._x = tf.placeholder(tf.float32, shape=input_shape, name='x')
        self._labels = tf.placeholder(tf.float32, shape=output_shape, name='labels')

        self._batch_size = tf.placeholder(tf.uint8, name='batch_size')
        self._learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        self._layers = hidden_layers or list()
        if params is not None:
            self._params = utils.AttrDict(**params)
        else:
            self._params = dict()

    def __repr__(self):
        return "<class 'poncoocr.model.CNN'" \
               "  name: {s._name}" \
               "  layers: {s._layers}>".format(s=self)

    @property
    def x(self):
        return self._x

    @property
    def labels(self):
        return self._labels

    @property
    def name(self):
        return self._name

    @property
    def hidden_layers(self):
        return tuple(self._layers)

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def learning_rate(self):
        return self._learning_rate

    @classmethod
    def from_architecture(cls, arch: architecture.CNNArchitecture):
        """Initialize the model from the given Architecture"""
        # load network parameters
        if arch.name == 'default':
            name = None
        else:
            name = arch.name

        batch_size = arch.batch_size
        learning_rate = arch.learning_rate
        optimizer = getattr(tf.train, arch.optimizer, tf.train.AdamOptimizer)
        model = cls(
            input_shape=arch.input_shape,
            output_shape=arch.output_shape,
            name=name
        )

        model.set_parameters(dict(batch_size=batch_size,
                                  learning_rat=learning_rate,
                                  optimizer=optimizer)
                             )

        # Load each layer from architecture and add it to the model
        input_layer = model.x
        for layer_data in arch.layers:
            # Construct a layer object
            layer = _layer_dict[layer_data.name](
                name=layer_data.name,
                input_data=input_layer,
                output_channels=layer_data.output_channels,
                activation=layer_data.activation,
                **layer_data.params
            )
            model.add_layer(layer)
            input_layer = layer

        return model

    def set_parameters(self, dct: dict):
        self._params = utils.AttrDict(**dct)

    def train(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def add_layer(self, layer: typing.Union[layers.ConvLayer, layers.FullyConnectedLayer]):
        """Add layer to the model."""
        # TODO: check if shapes do not agree
        if not self._layers and layer.input_data is None:
            raise AttributeError("The input layer must have `input_data` and `input_shape` specified.")

        if self._layers:
            layer.connect(self._layers[-1])

        self._layers.append(layer)
