"""Module containing estimator for poncoocr engine."""

import os
import tempfile
import typing

import numpy as np
import tensorflow as tf

from PIL import Image

from .architecture import ModelArchitecture
from .dataset import Dataset
from .model import Model


class Estimator(object):

    __model_cache_dir = '.model_cache/'

    def __init__(self,
                 model_architecture: ModelArchitecture,
                 train_data: Dataset,
                 test_data: Dataset=None,
                 model_fn=None,
                 model_dir=None,
                 params: dict = None):
        """Initialize estimator initializer with a model to be used for the estimator."""
        self._arch = model_architecture
        self._train_data = train_data
        self._test_data = test_data
        self._params = params or dict()

        if not os.path.isdir(self.__model_cache_dir):
            os.mkdir(self.__model_cache_dir)

        self._cache_dir = model_dir
        if self._cache_dir is None:
            self._cache_dir = tempfile.mkdtemp(dir=os.path.join(self.__model_cache_dir),
                                               prefix='arch_',
                                               suffix='_%s' % self._arch.name)
        else:
            self._cache_dir = os.path.join(self.__model_cache_dir, self._cache_dir)
            if not os.path.isdir(self._cache_dir):
                os.mkdir(path=self._cache_dir)

        self._estimator = self._get_estimator(model_fn=model_fn)

    @property
    def test_data(self):
        return self._test_data

    @property
    def train_data(self):
        return self._train_data

    @property
    def classes(self):
        return self._test_data.classes

    def train(self, num_epochs=1, buffer_size=None):
        """Train the estimator.

        :param num_epochs: number of epochs to be the estimator trained for.
        :param buffer_size: size of the buffer that is used for shuffling.
        If `None`, buffer size is chosen according to the size of the training data.
        """
        self._estimator.train(
            input_fn=lambda: self._dataset_input_fn(
                dataset=self._train_data,
                repeat=None,
                buffer_size=buffer_size or len(self._train_data)
            ),
            steps=len(self._train_data) / self._arch.batch_size * num_epochs
        )

    def evaluate(self) -> dict:
        """Evaluate accuracy of the estimator."""
        results = self._estimator.evaluate(
            input_fn=lambda: self._dataset_input_fn(
                dataset=self._test_data,
                repeat=1,
            ),
        )

        return results

    def predict(self, path: str) -> typing.Generator:
        """Produces class predictions about either a single image or multiple images.

        :param path: str, path to an image or directory of images.
        """
        predictions = self._estimator.predict(
            input_fn=self._predict_input_fn(path=path)
        )

        return predictions

    def _input_fn(self, features, labels, one_hot=False, depth=None, shuffle=False, num_epochs=None) -> tuple:
        """Input function for the estimator.
        Converts the features and labels into tensors and returns tuple ({'x': features}, {'labels': labels)."""

        if one_hot is True:
            if depth is None:
                tf.logging.warn(msg="`one_hot=True` but `depth` has not been provided."
                                    " Computed depth might not be correct.")
                depth = len(set(labels))  # If depth is not provided, attempt to compute it
            # Run a nested session to compute one-hot encoded labels
            with tf.Session() as _sess:
                labels = _sess.run(tf.one_hot(indices=labels, depth=depth))

        _fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': features},
            y=labels,
            batch_size=self._arch.batch_size,
            shuffle=shuffle,
            num_epochs=num_epochs
        )

        return _fn()

    def _dataset_input_fn(self, dataset: Dataset, repeat=None, buffer_size=None) -> tuple:
        """Input function for the estimator.
        Loads the dataset from a directory given by `path` and returns tuple ({'x': features}, {'labels': labels)."""
        buffer_size = buffer_size or len(dataset)

        iterator = dataset.make_one_shot_iterator(repeat=repeat, shuffle=buffer_size,
                                                  batch_size=self._arch.batch_size)
        # Use the graph invoked by estimator._train_model
        with tf.variable_scope('input_layer', reuse=True):
            features, labels = iterator.get_next()
            print(features, labels)

        return {'x': features}, labels

    def _predict_input_fn(self, path: str) -> dict:

        if os.path.isdir(path):
            dataset = Dataset.from_directory(path)

            dataset = dataset.batch(self._arch.batch_size)

            iterator = dataset.make_one_shot_iterator()
            # Use the graph invoked by estimator._train_model
            with tf.variable_scope('predict_input'):
                features, _ = iterator.get_next()

        else:
            # load and normalize the array
            img_arr = np.asarray(Image.open(path, 'r').convert('L')) / 255
            img_tensor = tf.convert_to_tensor(img_arr, dtype=tf.float32)
            img_tensor = tf.expand_dims(img_tensor, 0)

            dataset = tf.data.Dataset.from_tensor_slices(img_tensor)

            iterator = dataset.batch(self._arch.batch_size).make_one_shot_iterator()
            features = iterator.get_next()

        return {'x': features}

    def _get_estimator(self, model_fn=None):
        """Returns the estimator with the model function.
        :param model_fn: custom model function which will be passed to the Estimator. See `tf.estimator.Estimator`
        documentation for more info.
        """

        return tf.estimator.Estimator(
            model_fn=model_fn or self._model_fn,
            model_dir=self._cache_dir,
            params=self._params
        )

    def _model_fn(self, features, labels, mode, params=None) -> tf.estimator.EstimatorSpec:
        """Function used to be passed to an estimator and called upon train, eval or prediction.
        :returns: EstimatorSpec, a custom estimator specifications
        """

        tf.summary.image('images', features['x'], 6)

        model = Model.from_architecture(
            inputs=features,
            labels=labels,
            arch=self._arch,
            params=params,
        )

        logits = model.logits

        y_pred = tf.nn.softmax(logits)
        # get the corresponding class
        y_pred_cls = tf.argmax(y_pred, axis=1)

        if mode == tf.estimator.ModeKeys.PREDICT:
            spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={
                    'logits': logits,
                    'probabilities': y_pred,
                    'class_ids': y_pred_cls,
                }
            )

        else:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=model.labels,
                logits=y_pred
            )

            # reduces the cross entropy batch-tensor to a single number
            # used for optimization of the nn
            loss = tf.reduce_mean(cross_entropy)

            # optimizer for improving nn
            optimizer = model.optimizer()

            # get the tf op for single-step optimization
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step()
            )

            # define evaluation metrics - the classification accuracy
            # NOTE: assuming model labels to be one-hot encoded, therefore argmax needs to be performed first
            accuracy = tf.metrics.accuracy(labels=tf.argmax(model.labels, axis=1), predictions=y_pred_cls)
            tf.summary.scalar(name='accuracy', tensor=accuracy[1])

            metrics = {
                'accuracy': accuracy
            }

            spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                eval_metric_ops=metrics
            )

        return spec
