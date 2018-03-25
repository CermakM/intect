"""Tests for model module."""

import unittest

import tensorflow as tf

# The imports will need to be fixed to test installed version instead of the dev one
from . import config
from src import poncoocr as pcr


_dataset = pcr.dataset.Dataset.from_directory(config.TEST_DATASET_PATH)
_features, _labels = _dataset.make_one_shot_iterator().get_next()

_features, _labels = tf.stack(_features), tf.stack(_labels)


class TestModel(unittest.TestCase):

    def test_model_init(self):
        """Test that the model is initialized properly."""
        model = pcr.model.Model(inputs=_features, labels=_labels)

        self.assertIsInstance(model, pcr.model.Model)

    def test_model_for_random_names(self):
        """Test that all initialized models have random names."""
        names = set()
        # generate 5 same models and check their names differ
        for i in range(5):
            model = pcr.model.Model(inputs=_features, labels=_labels)

            self.assertTrue(model.name not in names)
            names.add(model.name)

        self.assertFalse(not names)

    def test_model_from_architecture(self):
        """Test that the model loads the architecture properly."""
        arch = pcr.architecture.ModelArchitecture.from_yaml(config.TEST_ARCHITECTURE_YAML)

        model = pcr.model.Model.from_architecture(
            inputs=_features, labels=_labels, arch=arch
        )

        self.assertIsInstance(model, pcr.model.Model)
        self.assertIsNotNone(model.name)
        self.assertFalse(not model.hidden_layers)

    def test_model_add_conv_layer(self):
        """Test adding layer to a model."""
        model = pcr.model.Model(inputs=_features, labels=_labels)
        self.assertTrue(not model.hidden_layers)
        model.add_conv_layer(
            filters=16,
            kernel_size=(3, 3),
            padding='same',
            activation=tf.nn.relu
        )

        self.assertFalse(not model.hidden_layers)

    def test_model_add_pool_layer(self):
        model = pcr.model.Model(inputs=_features, labels=_labels)
        self.assertTrue(not model.hidden_layers)
        model.add_max_pooling_layer(
            pool_size=(2, 2),
            strides=2,
        )

        self.assertFalse(not model.hidden_layers)

    def test_model_add_dense_layer(self):
        """Test adding layer to a model."""
        model = pcr.model.Model(inputs=_features, labels=_labels)
        self.assertTrue(not model.hidden_layers)
        model.add_dense_layer(
            units=64,
            activation=tf.nn.relu,
        )

        self.assertFalse(not model.hidden_layers)

    # def test_model_save(self):
    #     arch = pcr.architecture.ModelArchitecture.from_yaml(common.TEST_ARCHITECTURE_YAML)
    #
    #     model = pcr.model.Model.from_architecture(inputs=None, labels=None, arch=arch)
    #     model.save(fp=common.TEST_MODEL_PATH)
