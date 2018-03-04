"""Tests for model module."""

import unittest

import numpy as np
import tensorflow as tf

# The imports will need to be fixed to test installed version instead of the dev one
from . import common
from src import poncoocr as pcr


class TestModel(unittest.TestCase):

    def test_model_init(self):
        model = pcr.model.CNN(input_shape=(32, 32, 3), output_shape=(None, 62))
        self.assertIsInstance(model, pcr.model.CNN)

    def test_model_for_random_names(self):
        names = set()
        # generate 5 same models and check their names differ
        for i in range(5):
            model = pcr.model.CNN(input_shape=(None, 32, 32, 3), output_shape=(None, 62))
            self.assertTrue(model.name not in names)
            names.add(model.name)

        self.assertFalse(not names)

    def test_model_load_architecture(self):
        arch = pcr.architecture.CNNArchitecture.from_yaml(common.TEST_ARCHITECTURE_YAML)

        model = pcr.model.CNN.from_architecture(arch)

        self.assertIsInstance(model, pcr.model.CNN)
        self.assertIsNotNone(model.name)
        self.assertFalse(not model.hidden_layers)

    def test_model_add_layer(self):
        model = pcr.model.CNN(input_shape=(32, 32, 3), output_shape=[10])
        # Should raise, no input data provided yet
        with self.assertRaises(AttributeError):
            default_fcl = pcr.layers.FullyConnectedLayer(output_channels=1024)
            model.add_layer(layer=default_fcl)

        sess = tf.Session()  # TODO: make test graph namespace?
        dataset = pcr.dataset.Dataset.from_directory(common.TEST_DATASET_PATH)
        iterator = dataset.make_one_shot_iterator()
        features, labels = sess.run(iterator.get_next())

        default_fcl = pcr.layers.FullyConnectedLayer(input_data=np.array(features), output_channels=1024)
        model.add_layer(layer=default_fcl)

        self.assertFalse(not model.hidden_layers)

        # TODO: add more tests if shapes do not agree

    # def test_model_train(self):
    #     pass
    #
    # def test_model_predict(self):
    #     pass
    #
    # def test_model_save(self):
    #     pass
