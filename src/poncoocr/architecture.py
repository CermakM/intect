"""Module used to architect convolutional neural network."""

import json
import yaml

import typing

from . import utils


class ModelArchitecture(utils.AttrDict):
    """Class representing model architecture and hyper parameters."""

    def __init__(self,
                 name: str,
                 layers: typing.Sequence,
                 optimizer: str = 'AdamOptimizer',
                 batch_size: int = 32,
                 learning_rate: float = 1E-3,
                 **kwargs):
        """Initialize architecture of a Deep Neural Network."""
        # obligatory
        self.name = name or 'default'

        # optional
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer

        # initialize the layers as an attr dicts
        self.layers = [utils.AttrDict(**layer) for layer in layers]

        # load the rest of the values as attr dict
        super().__init__(**kwargs)

    @classmethod
    def from_json(cls, fp: str):
        """Loads the architecture from .json file."""
        with open(fp, 'r') as f:
            dct = json.load(f)

        return cls(**dct)

    @classmethod
    def from_yaml(cls, fp: str):
        """Loads the architecture from a .yaml file."""
        with open(fp, 'r') as f:
            dct = yaml.safe_load(f)

        return cls(**dct)

    @staticmethod
    def _dictionarize(dct: utils.AttrDict):
        _dct = dict()
        for k, v in dct.items():
            if isinstance(v, utils.AttrDict):
                _dct[k] = dict(v)
            else:
                _dct[k] = v

        return _dct

    def update(self, other):
        self.__dict__.update(other)

    def to_dict(self):
        """Dictionarize the architecture."""
        arch_dct = dict(self)
        arch_dct['layers'] = [self._dictionarize(v) for v in self.layers]

        return arch_dct

    def to_json(self, fp: str = None):
        if fp is None:
            return json.dumps(self.to_dict())

        with open(fp, 'w') as f:
            return json.dump(self.to_dict(), f)

    def to_yaml(self, fp: str = None):
        if fp is None:
            return yaml.safe_dump(self.to_dict())

        with open(fp, 'w') as f:
            f.write(yaml.safe_dump(self.to_dict()))

    def describe(self):
        return "<class 'poncoocr.architecture.ModelArchitecture'" \
               "  name: '{s.name}'" \
               "  layers: {s.layers}>".format(s=self)
