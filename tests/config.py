"""Module containing common variables and configurations for tests."""

import os
import tempfile


TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), 'data/')
TEST_DATASET_PATH = os.path.join(TEST_DATA_PATH, 'test_data/')

TEST_IMAGE_SAMPLE = os.path.join(TEST_DATA_PATH, 'default.png')

TEST_ARCHITECTURE_JSON = os.path.join(TEST_DATA_PATH, 'test-architecture.json')
TEST_ARCHITECTURE_YAML = os.path.join(TEST_DATA_PATH, 'test-architecture.yaml')

TEST_LOGDIR = tempfile.mkdtemp(prefix='tf_', suffix='_test')
