"""Tests for estimator module."""

import unittest

import tensorflow as tf
from PIL import Image

# The imports will need to be fixed to test installed version instead of the dev one
from . import common
from src import poncoocr as pcr

_dataset = pcr.dataset.Dataset.from_directory(common.TEST_DATASET_PATH).batch(32)
_features, _labels = _dataset.make_one_shot_iterator().get_next()

# Wrap features into a dict
_features = {'x': _features}


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

    def test_estimator_train(self):
        """Test training the model using estimator.
        The training should be done in at most 5 sec (based on the training data).
        """
        arch = pcr.architecture.ModelArchitecture.from_yaml(common.TEST_ARCHITECTURE_YAML)
        estim_initializer = pcr.estimator.EstimatorInitializer(model_architecture=arch)

        estimator = estim_initializer.get_estimator()
        # initialize deadline check -> raises if exceeded timeout
        thread = pcr.utils.Timeout(
            timeout=15, thread_id=1, name='deadline-check', func=estimator.train
        )
        thread.start()
        # perform 10 training steps
        estimator.train(
            input_fn=lambda: estim_initializer.input_fn(path=common.TEST_DATASET_PATH,
                                                        repeat=None,
                                                        buffer_size=10
                                                        ),
            steps=10
        )
        thread.stop()

    # def test_estimator_evaluate(self):
    #     """Test evaluating the model using estimator."""
    #     arch = pcr.architecture.ModelArchitecture.from_yaml(common.TEST_ARCHITECTURE_YAML)
    #     initializer = pcr.estimator.EstimatorInitializer(model_architecture=arch)
    #
    #     estimator = initializer.get_estimator()
    #     estimator.train(
    #         input_fn=lambda: initializer.input_fn(dataset=_dataset, repeat=1, buffer_size=10)
    #     )
    #     # Should run evaluation until `input_fn` raises StopIterration
    #     result = estimator.evaluate(
    #         input_fn=lambda: initializer.input_fn(dataset=_dataset, repeat=1, buffer_size=10)
    #     )
    #
    #     self.assertIsNotNone(result)

    # def test_estimator_predict(self):
    #     """Test prediction using estimator."""
    #     arch = pcr.architecture.ModelArchitecture.from_yaml(common.TEST_ARCHITECTURE_YAML)
    #     initializer = pcr.estimator.EstimatorInitializer(model_architecture=arch)
    #
    #     estimator = initializer.get_estimator()
    #
    #     # Load sample image
    #     image = Image.open(common.TEST_IMAGE_SAMPLE)
    #     predictions = estimator.predict()
    #
    #     self.assertIsNotNone(predictions)
