"""Module containing model of convolutional neural network for the poncoocr engine."""

import typing

import names
import tensorflow as tf

from tensorflow.python.framework import ops

from . import architecture


class Model(object):

    __model_names = set()

    def __init__(self,
                 inputs,
                 labels,
                 name: str = None,
                 params: dict = None):
        """Initialize the model."""

        if name in self.__model_names:
            # TODO: allow? reuse?
            # raise ValueError("Model name `%s` already exists." % name)
            pass

        if name is None:
            # Generate some random name
            name = names.get_first_name(gender='female')
            while name in self.__model_names:
                name = names.get_first_name(gender='female')

        self._name = name
        self.__model_names.add(self._name)

        # noinspection PyProtectedMember
        self._graph = ops._get_graph_from_inputs([inputs, labels])  # pylint: disable=protected-access
        self._graph_context_manager = self._graph.as_default()
        self._graph_context_manager.__enter__()

        with tf.variable_scope('input_layer'):
            self._x = inputs['x']

        with tf.variable_scope('labels'):
            self._labels = labels

        self._layers = [self._x]
        self._layer_name_dct = dict()
        "Dictionary of ([str]layer_name, [int]count)"

        if params is None:
            params = dict()

        # Configurable and directly accessible properties
        for param, default in [('batch_size', None), ('learning_rate', None)]:
            # Make sure that flags are privileged from yaml arguments
            value = tf.app.flags.FLAGS.get_flag_value(param, default)
            if value is None:
                try:
                    value = params.get(param)
                except KeyError:
                    raise("attribute `{}` cannot be of type {}.".format(param, type(None)))

            self.__setattr__(param, tf.constant(value, tf.float32))

        _decay_dct = params.get('learning_rate_decay', None)
        if isinstance(_decay_dct, list):  # yaml will pass list here, need to unpack
            _decay_dct, = _decay_dct

        self.learning_rate_decay = _decay_dct is not None
        "bool"

        if self.learning_rate_decay:
            self._decay_steps = tf.constant(_decay_dct.get('decay_steps'), tf.float32)
            self._decay_rate = tf.constant(_decay_dct.get('decay_rate'), tf.float32)

            self.learning_rate = tf.train.exponential_decay(learning_rate=self.learning_rate,
                                                            global_step=tf.train.get_global_step(),
                                                            decay_steps=self._decay_steps,
                                                            decay_rate=self._decay_rate,
                                                            )

        tf.summary.scalar(tensor=self.learning_rate, name='learning_rate')

        self.optimizer = getattr(tf.train, params.get('optimizer', 'AdamOptimizer'))
        "tf.train.Optimizer"

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._graph_context_manager.__exit__(exc_type, exc_val, exc_tb)

    @property
    def graph(self):
        return self._graph or tf.get_default_graph()

    @property
    def x(self):
        return self._x

    @property
    def labels(self):
        return self._labels

    @property
    def input_layer(self):
        return self._layers[0]

    @property
    def hidden_layers(self):
        return tuple(self._layers[1:-1])

    @property
    def logits(self):
        return self._layers[-1]

    @property
    def name(self):
        return self._name

    @classmethod
    def from_architecture(cls, inputs, labels, arch: architecture.ModelArchitecture, params: dict = None):
        """Initialize the model from the given Architecture"""
        # load network parameters
        config = arch.get_configurations()

        if arch.name == 'default':
            name = None
        else:
            name = arch.name

        if params:
            config.update(params)

        model = cls(
            inputs=inputs,
            labels=labels,
            name=name,
            params=config,
        )

        # Load each layer from architecture and add it to the model
        for i, layer in enumerate(arch.layers):
            # Construct a layer by the type specified in the architecture
            config = layer.params or {}

            if i < len(arch.layers) - 1:
                scope = 'hidden_layer_%d' % i
            else:
                scope = 'output_layer'

            model.add_layer(layer_type=layer.type, scope=scope, name=layer.name, **config)

        return model

    def add_layer(self, layer_type, scope=None, *args, **kwargs):
        """Add layer specified by `layer_type` argument to the model."""
        assert isinstance(layer_type, str), "expected argument `layer_type` of type `%s`" % type(str)

        if scope is None:
            scope = 'layer_%d' % len(self.hidden_layers)

        with tf.variable_scope(scope):

            if layer_type == 'conv2d':
                self.add_conv_layer(*args, **kwargs)

            elif layer_type == 'max_pooling2d':
                self.add_max_pooling_layer(*args, **kwargs)

            elif layer_type == 'flatten':
                self.add_flatten_layer()

            elif layer_type == 'dense':
                self.add_dense_layer(*args, **kwargs)

            else:
                raise AttributeError("Invalid argument `layer_type` provided: `%s`" % layer_type)

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
        layer_name = self.get_unique_layer_name(name or kwargs.get('name', 'conv'))

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

        # add tensorboard summaries
        tf.summary.scalar(name='weights', tensor=tf.reduce_mean(conv))
        tf.summary.scalar(name='sparsity', tensor=tf.nn.zero_fraction(conv))
        tf.summary.histogram(name=layer_name, values=conv)

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
        layer_name = self.get_unique_layer_name(name or kwargs.get('name', 'dense'))

        dense = tf.layers.dense(
            inputs=self._layers[-1],
            units=units,
            activation=activation,
            name=layer_name,
            *args, **kwargs
        )

        # add tensorboard summaries
        tf.summary.scalar(name='weights', tensor=tf.reduce_mean(dense))
        tf.summary.scalar(name='sparsity', tensor=tf.nn.zero_fraction(dense))
        tf.summary.histogram(name=layer_name, values=dense)

        self._layers.append(dense)

    def add_max_pooling_layer(self,
                              pool_size: typing.Union[typing.Sequence, tf.TensorShape],
                              strides: int = 2,
                              name=None,
                              *args, **kwargs):

        # Uniquify layer name
        layer_name = self.get_unique_layer_name(name or kwargs.get('name', 'pool'))

        pool = tf.layers.max_pooling2d(
            inputs=self._layers[-1],
            pool_size=pool_size,
            strides=strides,
            name=layer_name,
            *args, **kwargs
        )

        # add tensorboard summaries
        tf.summary.scalar(name='weights', tensor=tf.reduce_mean(pool))
        tf.summary.scalar(name='sparsity', tensor=tf.nn.zero_fraction(pool))
        tf.summary.histogram(name=layer_name, values=pool)

        self._layers.append(pool)

    def save(self):
        raise NotImplementedError

    def get_unique_layer_name(self, name: str):
        if name in self._layer_name_dct:
            layer_name = "{name}_{id}".format(name=name, id=self._layer_name_dct[name])
            self._layer_name_dct[name] += 1
        else:
            layer_name = name
            self._layer_name_dct[name] = 1

        return layer_name

