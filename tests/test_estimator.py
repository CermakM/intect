"""Tests for estimator module."""

import unittest

import tensorflow as tf

import poncoocr as pcr
from . import config


# load the dataset here so that it doesn't need to be loaded at every test
_DATASET = pcr.dataset.Dataset.from_directory(config.TEST_DATASET_PATH)
_FEATURES, _LABELS = _DATASET.make_one_shot_iterator(batch_size=32).get_next()

# Wrap features into a dict
_FEATURES = {'x': _FEATURES}


class TestEstimator(unittest.TestCase):
    """Tests for estimator class."""

    def test_estimator_initializer(self):
        """Test estimator creation from a given model."""
        arch = pcr.architecture.ModelArchitecture.from_yaml(config.TEST_ARCHITECTURE_YAML)
        estimator = pcr.estimator.Estimator(train_data=_DATASET,
                                            model_architecture=arch)

        self.assertIsInstance(estimator, pcr.estimator.Estimator)

    def test_estimator_model_fn(self):
        """Test that estimator model function returns EstimatorSpec correctly for every mode."""
        arch = pcr.architecture.ModelArchitecture.from_yaml(config.TEST_ARCHITECTURE_YAML)
        estimator = pcr.estimator.Estimator(train_data=_DATASET,
                                            model_architecture=arch)

        modes = ['TRAIN', 'EVAL', 'PREDICT']
        for mode in modes:
            # create an estimator for each mode using the same model but with different
            # variable scopes - model layer scopes must not interfere
            with tf.variable_scope(mode):
                estimator_spec = estimator._model_fn(features=_FEATURES,
                                                     labels=_LABELS,
                                                     mode=getattr(tf.estimator.ModeKeys, mode),
                                                     )

            self.assertIsInstance(estimator_spec, tf.estimator.EstimatorSpec)

    def test_estimator_train(self):
        """Test training the model using estimator.
        The training should be done in at most 5 sec (based on the training data).
        """
        arch = pcr.architecture.ModelArchitecture.from_yaml(config.TEST_ARCHITECTURE_YAML)
        estimator = pcr.estimator.Estimator(train_data=_DATASET,
                                            model_architecture=arch)

        # initialize deadline check -> raises if exceeded timeout
        thread = pcr.utils.Timeout(
            timeout=60, thread_id=7, name='deadline-check', func_name='estimator.train'
        )
        thread.start()

        # perform 10 training steps in the time given
        estimator.train(steps=10)

        thread.stop()

    def test_estimator_evaluate(self):
        """Test evaluating the model using estimator."""
        arch = pcr.architecture.ModelArchitecture.from_yaml(config.TEST_ARCHITECTURE_YAML)
        estimator = pcr.estimator.Estimator(train_data=_DATASET,
                                            model_architecture=arch)

        estimator.train(steps=10),

        # Evaluation should run until the `input_fn` raises StopIteration
        thread = pcr.utils.Timeout(
            timeout=5, thread_id=7, name='deadline-check', func_name='estimator.train'
        )
        thread.start()
        # run evaluation once
        results = estimator.evaluate(test_data=_DATASET)
        thread.stop()

        self.assertIsInstance(results, dict)
        # getattr is not defined on
        self.assertIsInstance(results.get('accuracy', None), tf.float32.as_numpy_dtype)

    def test_estimator_predict(self):
        """Test prediction using estimator."""
        arch = pcr.architecture.ModelArchitecture.from_yaml(config.TEST_ARCHITECTURE_YAML)
        estimator = pcr.estimator.Estimator(train_data=_DATASET,
                                            model_architecture=arch)

        estimator.train(train_data=_DATASET, steps=10),

        # Test prediction of a single image
        predictions = estimator.predict(fp=[config.TEST_IMAGE_SAMPLE])

        # check that predictions is not an empty list
        self.assertFalse(not list(predictions))

        # Test prediction of a list of images
        num_samples = 5
        predictions = estimator.predict(fp=[config.TEST_IMAGE_SAMPLE] * num_samples)
        predictions = list(predictions)

        # check that predictions is not an empty list
        self.assertFalse(not predictions)
        # check that each sample has corresponding prediction
        self.assertEqual(len(predictions), num_samples)
