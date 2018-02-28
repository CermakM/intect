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
    pass
