"""Tests for architecture module."""

import typing
import unittest

# The imports will need to be fixed to test installed version instead of the dev one
from . import common
from src import poncoocr as pcr


class TestArchitecture(unittest.TestCase):

    def test_architecture_from_json(self):
        arch = pcr.architecture.CNNArchitecture.from_json(common.TEST_ARCHITECTURE_JSON)

        self.assertIsInstance(arch, pcr.architecture.CNNArchitecture)

    def test_architecture_from_yml(self):
        arch = pcr.architecture.CNNArchitecture.from_yaml(common.TEST_ARCHITECTURE_YAML)

        self.assertIsInstance(arch, pcr.architecture.CNNArchitecture)

    def test_architecture_to_json(self):
        import json
        arch = pcr.architecture.CNNArchitecture.from_json(common.TEST_ARCHITECTURE_JSON)
        with open(common.TEST_ARCHITECTURE_JSON) as f:
            arch_json = json.load(f)
        self.assertEqual(json.loads(arch.to_json()), arch_json)

    def test_architecture_to_yaml(self):
        import yaml
        arch = pcr.architecture.CNNArchitecture.from_yaml(common.TEST_ARCHITECTURE_YAML)
        with open(common.TEST_ARCHITECTURE_YAML) as f:
            arch_json = yaml.load(f)
        self.assertEqual(arch_json, arch.to_yaml())
        pass
