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
                 input_shape: typing.Sequence,
                 output_shape: typing.Sequence,
                 optimizer: str = 'adam',
                 batch_size: int = 32,
                 learning_rate: float = 1E-3,
                 **kwargs,
                 ):
        """Initialize architecture of a Convolutional Neural Network."""
        # obligatory
        self.name = name
        self.layers = layers

        self.input_shape = input_shape
        self.output_shape = output_shape

        # optional
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer

        # load the rest of the values as attrdict
        super().__init__(**kwargs)

    @classmethod
    def from_json(cls, fp: str):
        """Loads the architecture from .json file."""
        with open(fp, 'r') as f:
            dct = json.load(f)

        name = dct.pop('name')
        return cls(name, **dct)

    @classmethod
    def from_yaml(cls, fp: str):
        """Loads the architecture from a .yaml file.
        Note: the architecture.yaml has to be in ``safe_load`` format
        in order to assure correctly loaded architecture.
        """
        with open(fp, 'r') as f:
            dct = yaml.safe_load(f)

        return cls(**dct)

    def to_json(self, fp: str = None):
        if fp is None:
            return json.dumps(self.__dict__)

        with open(fp, 'w') as f:
            return json.dump(self.__dict__, f)

    def to_yaml(self, fp: str = None):
        if fp is None:
            return yaml.safe_dump(self.__dict__)

        with open(fp, 'w') as f:
            f.write(yaml.safe_dump(self.__dict__))

    def describe(self):
        return self.__dict__.__str__()
