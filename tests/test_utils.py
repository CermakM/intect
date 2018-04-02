"""Tests for utils module."""

import os
import time
import unittest

import numpy as np

import poncoocr as pcr
from . import config


class TestUtils(unittest.TestCase):

    def test_utils_attribute_dict(self):
        """Test AttrDict."""
        # test accessibility of attributes from dict
        attr_dct = pcr.utils.AttrDict(**{'default_key': 'default_value', 'dashed-key': 'dashed-value'})

        # noinspection PyUnresolvedReferences
        self.assertTrue(attr_dct.default_key == 'default_value')  # pylint: disable=no-member
        # noinspection PyUnresolvedReferences
        self.assertTrue(attr_dct.dashed_key == 'dashed-value')  # pylint: disable=no-member

    def test_utils_timeout_stop(self):
        """Test timeout interruption."""
        timeout = 2
        thread = pcr.utils.Timeout(timeout=2, thread_id=1, name='test-deadline')
        thread.start()

        self.assertTrue(thread.is_alive())

        thread.stop()
        time.sleep(timeout)  # Let the thread terminate properly and check it did not raise

        self.assertFalse(thread.is_alive())

    def test_make_hparam_string(self):
        """Test hyper parameter string creation from architecture spec."""

        arch = pcr.architecture.ModelArchitecture.from_yaml(config.TEST_ARCHITECTURE_YAML)
        string = pcr.utils.make_hparam_string(arch)

        self.assertEqual(string, "{name},lr={lr},bs={bs},conv=1,fcl=1".format(
            name=arch.name,
            bs=arch.batch_size,
            lr=arch.learning_rate,
        ))

    def test_make_sprite_image(self):
        dataset = pcr.dataset.Dataset.from_directory(config.TEST_DATASET_PATH)
        features, labels = dataset.features, np.argmax(dataset.labels, axis=1)

        sprite, meta = pcr.utils.make_sprite_image(
            images=features, metadata=labels, num_images=100, dir_path=config.TEST_LOGDIR)

        # check that sprite.png and metadata.tsv have been created
        files = set(os.listdir(config.TEST_LOGDIR))
        print(sprite, meta, files)
        self.assertTrue(os.path.basename(sprite) in files)
        self.assertTrue(os.path.basename(meta) in files)

    def test_label_to_class(self):
        dataset = pcr.dataset.Dataset.from_directory(config.TEST_DATASET_PATH)
        classes = pcr.utils.label_to_class(
            labels=dataset.labels,
            class_dct=dataset.classes
        )

        self.assertIsNotNone(classes)
        self.assertEqual(len(classes), len(dataset))

