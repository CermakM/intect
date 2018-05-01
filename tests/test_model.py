"""Tests for model module."""

import unittest

import tensorflow as tf

import intect
from . import config


_DATASET = intect.dataset.Dataset.from_directory(config.TEST_DATASET_PATH)
_FEATURES, _LABELS = _DATASET.make_one_shot_iterator(batch_size=32).get_next()

# wrap the features into a dict
_FEATURES = {'x': _FEATURES}

# mandatory parameters (which are usually passed by tf.app)
PARAMS = {
    'batch_size': 32,
    'learning_rate': 1E-4
}


class TestModel(unittest.TestCase):
    """Tests for Model class."""

    def test_model_init(self):
        """Test that the model is initialized properly."""
        model = intect.model.Model(inputs=_FEATURES, labels=_LABELS, params=PARAMS)

        self.assertEqual(len(model._layers), 1)  # pylint: disable=protected-access

        self.assertIsInstance(model.input_layer, tf.Tensor)
        self.assertIsInstance(model, intect.model.Model)

    def test_model_for_random_names(self):
        """Test that all initialized models have random names."""
        names = set()
        # generate 5 same models and check their names differ
        for _ in range(5):
            model = intect.model.Model(inputs=_FEATURES, labels=_LABELS, params=PARAMS)

            self.assertTrue(model.name not in names)
            names.add(model.name)

        self.assertFalse(not names)

    def test_model_from_architecture(self):
        """Test that the model loads the architecture properly."""
        arch = intect.architecture.ModelArchitecture.from_yaml(config.TEST_ARCHITECTURE_YAML)

        model = intect.model.Model.from_architecture(
            inputs=_FEATURES, labels=_LABELS, arch=arch
        )

        self.assertIsInstance(model, intect.model.Model)
        self.assertIsNotNone(model.name)
        self.assertFalse(not model.hidden_layers)

    def test_model_add_conv_layer(self):
        """Test adding layer to a model."""
        model = intect.model.Model(inputs=_FEATURES, labels=_LABELS, params=PARAMS)
        self.assertEqual(len(model._layers), 1)  # pylint: disable=protected-access
        model.add_conv_layer(
            filters=16,
            kernel_size=(3, 3),
            padding='same',
            activation=tf.nn.relu
        )

        self.assertEqual(len(model._layers), 2)  # pylint: disable=protected-access

    def test_model_add_dense_layer(self):
        """Test adding layer to a model."""
        model = intect.model.Model(inputs=_FEATURES, labels=_LABELS, params=PARAMS)
        self.assertEqual(len(model._layers), 1)  # pylint: disable=protected-access
        model.add_dense_layer(
            units=64,
            activation=tf.nn.relu,
        )

        self.assertEqual(len(model._layers), 2)  # pylint: disable=protected-access

    def test_model_add_pool_layer(self):
        """Test adding pooling layer to a model."""
        model = intect.model.Model(inputs=_FEATURES, labels=_LABELS, params=PARAMS)
        self.assertEqual(len(model._layers), 1)  # pylint: disable=protected-access
        model.add_max_pooling_layer(
            pool_size=(2, 2),
            strides=2,
        )

        self.assertEqual(len(model._layers), 2)  # pylint: disable=protected-access

    # def test_model_save(self):
    #     arch = intect.architecture.ModelArchitecture.from_yaml(common.TEST_ARCHITECTURE_YAML)
    #
    #     model = intect.model.Model.from_architecture(inputs=None, labels=None, arch=arch)
    #     model.save(fp=common.TEST_MODEL_PATH)
