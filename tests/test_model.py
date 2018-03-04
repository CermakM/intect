"""Tests for model module."""

from PIL import Image
import unittest

# The imports will need to be fixed to test installed version instead of the dev one
from . import common
from src import poncoocr as pcr


class TestModel(unittest.TestCase):

    def test_model_init(self):
        """Test that the model is initialized properly."""
        model = pcr.model.Model(input_shape=(32, 32, 3), output_shape=(None, 62))
        self.assertIsInstance(model, pcr.model.Model)

    def test_model_for_random_names(self):
        """Test that all initialized models have random names."""
        names = set()
        # generate 5 same models and check their names differ
        for i in range(5):
            model = pcr.model.Model(input_shape=(None, 32, 32, 3), output_shape=(None, 62))
            self.assertTrue(model.name not in names)
            names.add(model.name)

        self.assertFalse(not names)

    def test_model_load_architecture(self):
        """Test that the model loads the architecture properly."""
        arch = pcr.architecture.ModelArchitecture.from_yaml(common.TEST_ARCHITECTURE_YAML)

        model = pcr.model.Model.from_architecture(arch)

        self.assertIsInstance(model, pcr.model.Model)
        self.assertIsNotNone(model.name)
        self.assertFalse(not model.hidden_layers)

    def test_model_add_layer(self):
        """Test adding layer to a model."""
        model = pcr.model.Model(input_shape=(32, 32, 3), output_shape=[10])
        default_fcl = pcr.layers.FullyConnectedLayer(output_channels=1024)

        model.add_layer(layer=default_fcl)
        self.assertFalse(not model.hidden_layers)

        # TODO: add more tests if shapes do not agree

    def test_model_save(self):
        pass
