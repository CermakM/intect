"""Module containing common variables and configurations for tests."""

import os


TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
TEST_DATASET_PATH = os.path.join(TEST_DATA_PATH, 'test_data/')

TEST_IMAGE_SAMPLE = os.path.join(TEST_DATA_PATH, 'default.png')

TEST_ARCHITECTURE_JSON = os.path.join(TEST_DATA_PATH, 'default-architecture.json')
TEST_ARCHITECTURE_YAML = os.path.join(TEST_DATA_PATH, 'default-architecture.yaml')

TEST_MODEL_PATH = os.path.join(TEST_DATA_PATH, 'default-model')
