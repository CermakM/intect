"""Tests for estimator module."""

import unittest

import tensorflow as tf
from PIL import Image

# The imports will need to be fixed to test installed version instead of the dev one
from . import common
from src import poncoocr as pcr

_dataset = pcr.dataset.Dataset.from_directory(common.TEST_DATASET_PATH)
_features, _labels = _dataset.make_one_shot_iterator().get_next()

_features, _labels = tf.stack(_features), tf.stack(_labels)


class TestEstimator(unittest.TestCase):

    def test_estimator_initializer(self):
        """Test estimator creation from a given model."""
        arch = pcr.architecture.ModelArchitecture.from_yaml(common.TEST_ARCHITECTURE_YAML)
        initializer = pcr.estimator.EstimatorInitializer(model_architecture=arch)

        estimator = initializer.get_estimator()

        self.assertIsInstance(estimator, tf.estimator.Estimator)

    def test_estimator_model_fn(self):
        """Test that estimator model function returns EstimatorSpec correctly for every mode."""
        arch = pcr.architecture.ModelArchitecture.from_yaml(common.TEST_ARCHITECTURE_YAML)
        initializer = pcr.estimator.EstimatorInitializer(model_architecture=arch)

        modes = ['TRAIN', 'EVAL', 'PREDICT']
        for mode in modes:
            # create an estimator for each mode using the same model but with different
            # variable scopes
            estimator_spec = initializer.model_fn(features=_features,
                                                  labels=_labels,
                                                  mode=getattr(tf.estimator.ModeKeys, mode),
                                                  )

            self.assertIsInstance(estimator_spec, tf.estimator.EstimatorSpec)

    # def test_estimator_train(self):
    #     """Test training the model using estimator.
    #     The training should be done in at most 5 sec (based on the training data).
    #     """
    #     arch = pcr.architecture.ModelArchitecture.from_yaml(common.TEST_ARCHITECTURE_YAML)
    #     initializer = pcr.estimator.EstimatorInitializer(model_architecture=arch)
    #
    #     estimator = initializer.get_estimator()
    #     # initialize deadline check
    #     thread = pcr.utils.Timeout(timeout=5, thread_id=1, name='deadline-check',
    #                                func=estimator.train)
    #     thread.start()
    #     estimator.train(input_fn=lambda: initializer.input_fn())
    #     thread.stop()

    # def test_estimator_evaluate(self):
    #     """Test evaluating the model using estimator."""
    #     ...
    #     result = estimator.evaluate(input_fn=pcr.estimator.test_input_fn)
    #
    #     self.assertIsNotNone(result)
    #
    # def test_estimator_predict(self):
    #     """Test prediction using estimator."""
    #     # Load sample image
    #     image = Image.open(common.TEST_IMAGE_SAMPLE)
    #     # TODO
    #
    #     ...
    #     predictions = estimator.predict(input_fn=pcr.estimator.predict_input_fn)
    #
    #     self.assertIsNotNone(predictions)
