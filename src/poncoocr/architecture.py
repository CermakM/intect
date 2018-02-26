"""Module used to architect convolutional neural network."""

import json
import yaml

import typing

from . import utils


class CNNArchitecture(utils.AttrDict):
    """Class representing CNN architecture and hyper parameters."""

    def __init__(self,
                 name: str,
                 layers: typing.Sequence,
                 activation: typing.Union[typing.Sequence, str],
                 filters: typing.Sequence[int],
                 filter_shape: typing.Union[typing.Sequence[tuple], tuple],
                 input_shape: typing.Sequence,
                 batch_size: int = 32,
                 learning_rate: float = 1E-3,
                 optimizer: str = 'adam',
                 stride: int = 1,
                 padding: str = "SAME",
                 **kwargs,
                 ):
        """Initialize architecture of a Convolutional Neural Network."""
        # obligatory arguments
        self.name = name

        # If one activation is proved, perhaps user means to have the same activation function for all layers.
        #  also handle case if activations is of str type
        if len(activation) == 1 or isinstance(activation, str):
            activation = [activation] * len(layers)

        if not len(activation) == len(layers):
            raise AttributeError("Number of ``layers`` does not match the number of ``activation``. %d != %d" %
                                 (len(activation), len(layers)))

        self.layers = layers
        self.activation = activation

        # If one filter_shape is proved, perhaps user means to have the same filter_shape function for all filters.
        #  also handle case if filter_shape is of tuple type
        if len(filter_shape) == 1 or isinstance(filter_shape, tuple):
            filter_shape = [filter_shape] * len(filters)

        if not len(filters) == len(filter_shape):
            raise AttributeError("Number of ``layers`` does not match the number of ``activation``. %d != %d" %
                                 (len(filters), len(filter_shape)))

        self.filters = filters
        self.filter_shape = filter_shape

        self.input_shape = input_shape

        # optional
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.stride = stride
        self.padding = padding

        # load the rest of the values as attrdict
        super().__init__(**kwargs)

    @classmethod
    def from_json(cls, fp: str):
        with open(fp, 'r') as f:
            dct = json.load(f)

        name = dct.pop('name')
        return cls(name, **dct)

    @classmethod
    def from_yaml(cls, fp: str):
        with open(fp, 'r') as f:
            dct = yaml.load(f)

        name = dct.pop('name')
        return cls(name, **dct)

    def to_json(self, fp: str = None):
        if fp is None:
            return json.dumps(self.__dict__)

        with open(fp, 'w') as f:
            return json.dump(self.__dict__, f)

    def to_yaml(self, fp: str = None):
        if fp is None:
            return yaml.dump(self.__dict__)

        with open(fp, 'w') as f:
            f.write(yaml.dump(self.__dict__))

    def describe(self):
        return self.__dict__.__str__()
