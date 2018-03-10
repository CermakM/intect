"""Tests for estimator module."""

import unittest

import tensorflow as tf

# The imports will need to be fixed to test installed version instead of the dev one
from . import config
from src import poncoocr as pcr

_dataset = pcr.dataset.Dataset.from_directory(config.TEST_DATASET_PATH).batch(32)
_features, _labels = _dataset.make_one_shot_iterator().get_next()

# Wrap features into a dict
_features = {'x': _features}


class TestEstimator(unittest.TestCase):

    def test_estimator_initializer(self):
        """Test estimator creation from a given model."""
        arch = pcr.architecture.ModelArchitecture.from_yaml(config.TEST_ARCHITECTURE_YAML)
        initializer = pcr.estimator.EstimatorInitializer(model_architecture=arch)

        estimator = initializer.get_estimator()

        self.assertIsInstance(estimator, tf.estimator.Estimator)

    def test_estimator_model_fn(self):
        """Test that estimator model function returns EstimatorSpec correctly for every mode."""
        arch = pcr.architecture.ModelArchitecture.from_yaml(config.TEST_ARCHITECTURE_YAML)
        initializer = pcr.estimator.EstimatorInitializer(model_architecture=arch)

        modes = ['TRAIN', 'EVAL', 'PREDICT']
        for mode in modes:
            # create an estimator for each mode using the same model but with different
            # variable scopes - model layer scopes must not interfere
            with tf.variable_scope(mode):
                estimator_spec = initializer._model_fn(features=_features,
                                                       labels=_labels,
                                                       mode=getattr(tf.estimator.ModeKeys, mode),
                                                       )

            self.assertIsInstance(estimator_spec, tf.estimator.EstimatorSpec)

    def test_estimator_train(self):
        """Test training the model using estimator.
        The training should be done in at most 5 sec (based on the training data).
        """
        arch = pcr.architecture.ModelArchitecture.from_yaml(config.TEST_ARCHITECTURE_YAML)
        estim_initializer = pcr.estimator.EstimatorInitializer(model_architecture=arch)

        estimator = estim_initializer.get_estimator()
        # initialize deadline check -> raises if exceeded timeout
        thread = pcr.utils.Timeout(
            timeout=10, thread_id=1, name='deadline-check', func_name='estimator.train'
        )
        thread.start()
        # perform 10 training steps
        estimator.train(
            input_fn=lambda: estim_initializer.input_fn(
                path=config.TEST_DATASET_PATH,
                repeat=None,
                buffer_size=10
            ),
            steps=10
        )
        thread.stop()

    def test_estimator_evaluate(self):
        """Test evaluating the model using estimator."""
        arch = pcr.architecture.ModelArchitecture.from_yaml(config.TEST_ARCHITECTURE_YAML)
        estim_initializer = pcr.estimator.EstimatorInitializer(model_architecture=arch)

        estimator = estim_initializer.get_estimator()

        estimator.train(
            input_fn=lambda: estim_initializer.input_fn(
                path=config.TEST_DATASET_PATH,
                repeat=None,
                buffer_size=10
            ),
            steps=10
        )

        # Evaluation should run until the `input_fn` raises StopIteration
        thread = pcr.utils.Timeout(
            timeout=5, thread_id=1, name='deadline-check', func_name='estimator.train'
        )
        thread.start()
        results = estimator.evaluate(
            input_fn=lambda: estim_initializer.input_fn(
                path=config.TEST_DATASET_PATH,
                repeat=5,
                buffer_size=10
            )
        )
        thread.stop()

        self.assertIsInstance(results, dict)
        # getattr is not defined on
        self.assertIsInstance(results.get('accuracy', None), tf.float32.as_numpy_dtype)

    def test_estimator_predict(self):
        """Test prediction using estimator."""
        arch = pcr.architecture.ModelArchitecture.from_yaml(config.TEST_ARCHITECTURE_YAML)
        estim_initializer = pcr.estimator.EstimatorInitializer(model_architecture=arch)

        estimator = estim_initializer.get_estimator()

        estimator.train(
            input_fn=lambda: estim_initializer.input_fn(
                path=config.TEST_DATASET_PATH,
                repeat=None,
                buffer_size=10
            ),
            steps=10
        )

        # Test predictions from a dir of images
        predictions = estimator.predict(
            input_fn=lambda: estim_initializer.predict_input_fn(
                path=config.TEST_DATASET_PATH
            )
        )
        # check that predictions is not an empty list
        self.assertFalse(not list(predictions))

        # Test prediction of a single image
        predictions = estimator.predict(
            input_fn=lambda: estim_initializer.predict_input_fn(
                path=config.TEST_IMAGE_SAMPLE
            )
        )

        # check that predictions is not an empty list
        self.assertFalse(not list(predictions))
