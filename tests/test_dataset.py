"""Tests for dataset module."""

import os
import unittest

import tensorflow as tf

# The imports will need to be fixed to test installed version instead of the dev one
from . import common
from src import poncoocr as pcr


class TestDirectoryIterator(unittest.TestCase):

    def test_directory_iterator(self):
        """Test the dataset is created properly from directory."""
        dir_iter = pcr.dataset.DirectoryIterator(directory=common.TEST_DATASET_PATH)
        dir_iter.describe()

        self.assertIsInstance(dir_iter, pcr.dataset.DirectoryIterator)
        self.assertFalse(dir_iter.empty, msg="DirectoryIterator is empty.")

    def test_directory_iterator_len(self):
        """Test if all the images are loaded into the dataset."""
        dir_iter = pcr.dataset.DirectoryIterator(directory=common.TEST_DATASET_PATH)
        dir_iter.describe()

        expected_len = 0
        for __, _, walkfiles in os.walk(common.TEST_DATASET_PATH):
            expected_len += sum(1 for _ in walkfiles)

        self.assertEqual(expected_len, len(dir_iter))

    def test_diretory_iterator_iter(self):
        """Test that the DirectoryIterator suports the `iter()` protocol."""
        dir_iter = pcr.dataset.DirectoryIterator(directory=common.TEST_DATASET_PATH)
        dir_iter.describe()

        iter(dir_iter)


class TestDataset(unittest.TestCase):

    def test_dataset_from_directory(self):
        """Test that dataset loaded successfully."""
        dataset = pcr.dataset.Dataset.from_directory(common.TEST_DATASET_PATH)
        self.assertIsInstance(dataset, tf.data.Dataset)
        # dataset is not empty
        self.assertGreater(len(dataset.output_shapes), 0)

    def test_dataset_features(self):
        """Test that the dataset contains correct feature tensors."""
        dataset = pcr.dataset.Dataset.from_directory(common.TEST_DATASET_PATH)
        pass

    def test_dataset_labels(self):
        """Test that the dataset contains correct label tensors."""
        # dataset = pcr.dataset.Dataset.from_directory(common.TEST_DATASET_PATH)
        pass
