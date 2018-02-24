"""Tests for dataset module."""

import os
import unittest

# The imports will need to be fixed to test installed version instead of the dev one
from . import common
from src import poncoocr as pcr


class TestDataset(unittest.TestCase):

    def test_dataset_from_directory(self):
        """Test the dataset is created properly from directory."""
        dataset = pcr.dataset.Dataset.from_directory(common.TEST_DATASET)
        dataset.describe()

        self.assertIsInstance(dataset, pcr.dataset.Dataset)
        self.assertFalse(dataset.empty, msg="Dataset is empty.")

    def test_dataset_len(self):
        """Test if all the images are loaded into the dataset."""
        dataset = pcr.dataset.Dataset.from_directory(common.TEST_DATASET)

        expected_len = 0
        for __, _, walkfiles in os.walk(common.TEST_DATASET):
            expected_len += sum(1 for _ in walkfiles)

        self.assertEqual(expected_len, len(dataset))
