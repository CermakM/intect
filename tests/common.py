"""Module containing common variables for tests."""

import os


TEST_DATASET_PATH = os.path.join(os.path.dirname(__file__), 'data/test_data/')
TEST_IMAGE_SAMPLE = os.path.join(TEST_DATASET_PATH, 'default.png')

TEST_ARCHITECTURE_JSON = os.path.join(os.path.dirname(__file__), 'data/default-architecture.json')
TEST_ARCHITECTURE_YAML = os.path.join(os.path.dirname(__file__), 'data/default-architecture.yaml')
