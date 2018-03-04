"""Tests for architecture module."""

import typing
import unittest

# The imports will need to be fixed to test installed version instead of the dev one
from . import common
from src import poncoocr as pcr


class TestArchitecture(unittest.TestCase):

    def test_architecture_from_json(self):
        """Test architecture initialization from JSON file."""
        arch = pcr.architecture.ModelArchitecture.from_json(common.TEST_ARCHITECTURE_JSON)

        self.assertIsInstance(arch, pcr.architecture.ModelArchitecture)
        self.assertIsInstance(arch.layers, list)
        self.assertFalse(not arch.layers)

    def test_architecture_from_yml(self):
        """Test architecture initialization from yml file."""
        arch = pcr.architecture.ModelArchitecture.from_yaml(common.TEST_ARCHITECTURE_YAML)

        self.assertIsInstance(arch, pcr.architecture.ModelArchitecture)
        self.assertIsInstance(arch.layers, list)
        self.assertFalse(not arch.layers)

    def test_architecture_to_json(self):
        """Test dumping architecture to JSON."""
        import json
        arch = pcr.architecture.ModelArchitecture.from_json(common.TEST_ARCHITECTURE_JSON)
        with open(common.TEST_ARCHITECTURE_JSON) as f:
            arch_json = json.load(f)
        self.assertEqual(json.loads(arch.to_json()), arch_json)

    def test_architecture_to_yaml(self):
        """Test dumping architecture to yml."""
        import yaml
        arch = pcr.architecture.ModelArchitecture.from_yaml(common.TEST_ARCHITECTURE_YAML)
        with open(common.TEST_ARCHITECTURE_YAML) as f:
            arch_dct = yaml.safe_load(f)
        self.assertEqual(arch_dct, yaml.safe_load(arch.to_yaml()))
