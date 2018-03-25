"""Tests for model module."""

import unittest

import tensorflow as tf

# The imports will need to be fixed to test installed version instead of the dev one
from . import config
from src import poncoocr as pcr


DATASET = pcr.dataset.Dataset.from_directory(config.TEST_DATASET_PATH)

# create data necessary to perform tests
FEATURES, LABELS = DATASET.features, DATASET.labels

SPRITE, METADATA = pcr.utils.make_sprite_image(
    images=FEATURES,
    metadata=LABELS,
    num_images=len(DATASET),
    thumbnail=DATASET.img_shape[:2],  # neglect the depth of images for thumbnail definition
    dir_path=config.TEST_LOGDIR,
)


class TestEmbeddingHook(unittest.TestCase):

    def test_embedding_hook_init(self):
        """Test EmbeddingHook initialization."""
        # initialize random tensor which will represent logits layer of a cnn
        embedding = tf.truncated_normal(shape=(32, 1024), stddev=0.1)
        # set up hook with custom parameters
        hook = pcr.session.EmbeddingHook(
            tensors=embedding,
            sprite=SPRITE,
            metadata=METADATA,
            logdir=config.TEST_LOGDIR,
        )
        self.assertIsInstance(hook, tf.train.SessionRunHook)

    def test_embedding_hook_begin(self):
        pass

    def test_embedding_hook_before_run(self):
        pass

    def test_embedding_hook_after_run(self):
        pass

    def test_embedding_hook(self):
        pass
