"""Tests for utils module."""

import time
import unittest

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

        self.assertEqual(string, "{name},lr={lr},bs={},conv=2,fcl=1".format(
            name=arch.name,
            bs=arch.batch_size,
            lr=arch.learning_rate,
        ))

