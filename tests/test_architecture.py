"""Tests for architecture module."""

import typing
import unittest

import intect

from . import config


class TestArchitecture(unittest.TestCase):
    """Tests for Architecture class"""

    def test_architecture_from_json(self):
        """Test architecture initialization from JSON file."""
        arch = intect.architecture.ModelArchitecture.from_json(config.TEST_ARCHITECTURE_JSON)

        self.assertIsInstance(arch, intect.architecture.ModelArchitecture)
        self.assertIsInstance(arch.layers, list)
        self.assertFalse(not arch.layers)

    def test_architecture_from_yml(self):
        """Test architecture initialization from yml file."""
        arch = intect.architecture.ModelArchitecture.from_yaml(config.TEST_ARCHITECTURE_YAML)

        self.assertIsInstance(arch, intect.architecture.ModelArchitecture)
        self.assertIsInstance(arch.layers, list)
        self.assertFalse(not arch.layers)

    def test_architecture_to_dict(self):
        arch = intect.architecture.ModelArchitecture.from_yaml(config.TEST_ARCHITECTURE_YAML)

        def check(k, val):
            self.assertNotIsInstance(val, intect.utils.AttrDict)

            if isinstance(val, typing.Iterable) and not isinstance(val, str):
                for v in val:
                    check(k, v)

            if isinstance(val, dict):
                for j, v in val.items():
                    check(j, v)

        for key, value in arch.to_dict().items():
            # check that all AttrDict classes have been dictionarized
            check(key, value)

    def test_architecture_to_json(self):
        """Test dumping architecture to JSON."""
        import json
        arch = intect.architecture.ModelArchitecture.from_json(config.TEST_ARCHITECTURE_JSON)
        with open(config.TEST_ARCHITECTURE_JSON) as f:
            real_dct = json.load(f)
        loaded_dump = json.loads(arch.to_json())

        self.assertEqual(real_dct, loaded_dump)

    def test_architecture_to_yaml(self):
        """Test dumping architecture to yml."""
        import yaml
        arch = intect.architecture.ModelArchitecture.from_yaml(config.TEST_ARCHITECTURE_YAML)
        with open(config.TEST_ARCHITECTURE_YAML) as f:
            real_dct = yaml.safe_load(f)
        loaded_dump = yaml.safe_load(arch.to_yaml())

        self.assertEqual(real_dct, loaded_dump)
