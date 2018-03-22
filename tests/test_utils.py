"""Tests for utils module."""

import os
import tempfile
import time
import unittest

import numpy as np

# The imports will need to be fixed to test installed version instead of the dev one
from . import config
from src import poncoocr as pcr


class TestUtils(unittest.TestCase):

    def test_utils_attribute_dict(self):
        """Test AttrDict."""
        # test accessibility of attributes from dict
        attr_dct = pcr.utils.AttrDict(**{'default_key': 'default_value', 'dashed-key': 'dashed-value'})

        self.assertTrue(attr_dct.default_key == 'default_value')
        self.assertTrue(attr_dct.dashed_key == 'dashed-value')

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

        self.assertEqual(string, "{name},lr={lr},bs={bs},conv=2,fcl=1".format(
            name=arch.name,
            bs=arch.batch_size,
            lr=arch.learning_rate,
        ))

    def test_make_sprite_image(self):
        dataset = pcr.dataset.Dataset.from_directory(config.TEST_DATASET_PATH, normalize=False)
        features, labels = dataset.features, np.argmax(dataset.labels, axis=1)

        _dir = tempfile.mkdtemp(prefix='test_')

        sprite, meta = pcr.utils.make_sprite_image(images=features, metadata=labels, num_images=100, dir_path=_dir)

        # check that sprite.png and metadata.tsv have been created
        files = set(os.listdir(_dir))
        print(sprite, meta, files)
        self.assertTrue(os.path.basename(sprite) in files)
        self.assertTrue(os.path.basename(meta) in files)
