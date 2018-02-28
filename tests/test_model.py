"""Tests for model module."""

import typing
import unittest

# The imports will need to be fixed to test installed version instead of the dev one
from . import common
from src import poncoocr as pcr


class TestModel(unittest.TestCase):

    def test_model_init(self):
        model =pcr.model.CNN()
        self.assertIsInstance(model, pcr.model.CNN)

    def test_model_from_architecture(self):
        arch = pcr.architecture.CNNArchitecture.from_yaml(common.TEST_ARCHITECTURE_YAML)
        model = pcr.model.CNN.from_architecture(arch=arch)
        self.assertIsInstance(model, pcr.model.CNN)

    def test_model_for_random_names(self):
        names = set()
        # generate 5 same models and check their names differ
        for i in range(5):
            model = pcr.model.CNN()
            self.assertTrue(model.name not in names)
            names.add(model.name)

    def test_model_add_layer(self):
        pass

    def test_model_train(self):
        pass

    def test_model_predict(self):
        pass

    def test_model_save(self):
        pass
