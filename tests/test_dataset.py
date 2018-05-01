"""Tests for dataset module."""

import os
import unittest

import numpy as np
import tensorflow as tf

import intect
from . import config


class TestDirectoryIterator(unittest.TestCase):
    """Tests for DirectoryIterator class."""

    def test_directory_iterator(self):
        """Test that the dataset is created properly from directory."""
        dir_iter = intect.dataset.DirectoryIterator(directory=config.TEST_DATASET_PATH)
        dir_iter.describe()

        self.assertIsInstance(dir_iter, intect.dataset.DirectoryIterator)
        self.assertFalse(dir_iter.empty, msg="DirectoryIterator is empty.")

    def test_directory_iterator_len(self):
        """Test if all the images are loaded into the dataset."""
        dir_iter = intect.dataset.DirectoryIterator(directory=config.TEST_DATASET_PATH)
        dir_iter.describe()

        expected_len = 0
        for __, _, walkfiles in os.walk(config.TEST_DATASET_PATH):
            expected_len += sum(1 for _ in walkfiles)

        self.assertEqual(expected_len, len(dir_iter))

    def test_diretory_iterator_iter(self):
        """Test that the DirectoryIterator suports the `iter()` protocol."""
        dir_iter = intect.dataset.DirectoryIterator(directory=config.TEST_DATASET_PATH)
        dir_iter.describe()

        iter(dir_iter)


class TestDataset(unittest.TestCase):
    """Tests for Dataset class."""

    def test_dataset_from_directory(self):
        """Test that dataset loaded successfully."""
        dataset = intect.dataset.Dataset.from_directory(config.TEST_DATASET_PATH)
        self.assertIsInstance(dataset, intect.dataset.Dataset)
        # dataset is not empty and contains features and labels
        self.assertTrue(dataset.features.any())
        self.assertTrue(dataset.labels.any())

    def test_dataset_iterator_values(self):
        """Test that the iterator produces correct shapes during tf.Session."""
        dataset = intect.dataset.Dataset.from_directory(config.TEST_DATASET_PATH)
        iterator = dataset.make_one_shot_iterator(batch_size=32)
        # iterate over the labels twice and check the shape
        with tf.Session() as sess:
            features, labels = sess.run(iterator.get_next())

        self.assertIsNotNone(features)
        self.assertIsNotNone(labels)
        features_shape, labels_shape = np.array(features).shape, np.array(labels).shape

        # 32 test images of shape (32, 32, 1), shape (10, 32, 32, 1)
        self.assertEqual(features_shape, (32, 32, 32, 1))
        # 32 labels, 2 classes, one-hot encoded -> shape (32, 2)
        self.assertEqual(labels_shape, (32, 2))
