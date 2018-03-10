"""Tests for dataset module."""

import os
import unittest

import numpy as np
import tensorflow as tf

# The imports will need to be fixed to test installed version instead of the dev one
from . import config
from src import poncoocr as pcr


class TestDirectoryIterator(unittest.TestCase):

    def test_directory_iterator(self):
        """Test that the dataset is created properly from directory."""
        dir_iter = pcr.dataset.DirectoryIterator(directory=config.TEST_DATASET_PATH)
        dir_iter.describe()

        self.assertIsInstance(dir_iter, pcr.dataset.DirectoryIterator)
        self.assertFalse(dir_iter.empty, msg="DirectoryIterator is empty.")

    def test_directory_iterator_len(self):
        """Test if all the images are loaded into the dataset."""
        dir_iter = pcr.dataset.DirectoryIterator(directory=config.TEST_DATASET_PATH)
        dir_iter.describe()

        expected_len = 0
        for __, _, walkfiles in os.walk(config.TEST_DATASET_PATH):
            expected_len += sum(1 for _ in walkfiles)

        self.assertEqual(expected_len, len(dir_iter))

    def test_diretory_iterator_iter(self):
        """Test that the DirectoryIterator suports the `iter()` protocol."""
        dir_iter = pcr.dataset.DirectoryIterator(directory=config.TEST_DATASET_PATH)
        dir_iter.describe()

        iter(dir_iter)


class TestDataset(unittest.TestCase):

    def test_dataset_from_directory(self):
        """Test that dataset loaded successfully."""
        dataset = pcr.dataset.Dataset.from_directory(config.TEST_DATASET_PATH)
        self.assertIsInstance(dataset, tf.data.Dataset)
        # dataset is not empty
        self.assertFalse(not dataset.output_shapes)

    def test_dataset_next(self):
        """Test that dataset returns correct iterator."""
        dataset = pcr.dataset.Dataset.from_directory(config.TEST_DATASET_PATH)

        self.assertIsInstance(dataset.batch(32), tf.data.Dataset)

        iterator = dataset.make_one_shot_iterator()
        _ = iterator.get_next()

    def test_dataset_iterator_values(self):
        """Test that the iterator produces correct shapes during tf.Session."""
        dataset = pcr.dataset.Dataset.from_directory(config.TEST_DATASET_PATH)
        dataset = dataset.shuffle(buffer_size=20).repeat(2).batch(5)
        iterator = dataset.make_one_shot_iterator()
        # iterate over the labels twice and check the shape
        with tf.Session() as sess:
            features, labels = sess.run(iterator.get_next())

        self.assertIsNotNone(features)
        self.assertIsNotNone(labels)
        features_shape, labels_shape = np.array(features).shape, np.array(labels).shape

        # 10 test images of shape (32, 32, 3), repeated 2 -> shape (10, 2, 32, 32, 3)
        self.assertEqual(features_shape, (10, 2, 32, 32, 3))
        # 10 labels for the images repeated 2 times, 5 classes -> shape (10, 2, 5)
        self.assertEqual(labels_shape, (10, 2, 5))
