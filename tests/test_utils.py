"""Tests for utils module."""

import typing
import unittest

# The imports will need to be fixed to test installed version instead of the dev one
from . import common
from src import poncoocr as pcr


class TestAttrDict(unittest.TestCase):

    def test_architecture_attribute_dict(self):
        # test accessibility of attributes from dict
        attr_dct = pcr.utils.AttrDict(**{'default_key': 'default_value', 'dashed-key': 'dashed-value'})

        self.assertTrue(attr_dct.default_key == 'default_value')
        self.assertTrue(attr_dct.dashed_key == 'dashed-value')


class TestUtils(unittest.TestCase):

    def test_utils_images_from_directory(self):
        images = pcr.utils.images_from_directory(common.TEST_DATASET_PATH)
        self.assertIsInstance(images, list)
        self.assertGreater(len(images), 0)
        for img in images:
            self.assertIsInstance(img, typing.Iterable)

    def test_utils_labels_from_directory(self):
        labels = pcr.utils.labels_from_directory(common.TEST_DATASET_PATH)
        self.assertIsInstance(labels, list)
        self.assertGreater(len(labels), 0)

    def test_utils_load_data_from_directory(self):
        data = pcr.utils.load_data_from_directory(common.TEST_DATASET_PATH)
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

    def test_utils_flow_from_directory(self):
        images, labels = pcr.utils.flow_from_directory(common.TEST_DATASET_PATH)
        self.assertIsInstance(images, typing.Generator)
        self.assertIsInstance(labels, typing.Generator)
