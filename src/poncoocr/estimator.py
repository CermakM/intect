"""Module containing estimator for poncoocr engine."""

import os
import tempfile

import numpy as np
import tensorflow as tf

from PIL import Image

from .architecture import ModelArchitecture
from .dataset import Dataset
from .model import Model


class EstimatorInitializer(object):

    __model_cache_dir = '.model_cache/'

    def __init__(self, model_architecture: ModelArchitecture, params: dict = None):
        """Initialize estimator initializer with a model to be used for the estimator."""

        self._arch = model_architecture
        self._params = params or dict()

        if not os.path.isdir(self.__model_cache_dir):
            os.mkdir(self.__model_cache_dir)

        self._arch_cache_dir = tempfile.mkdtemp(dir=os.path.join(self.__model_cache_dir),
                                                prefix='arch_',
                                                suffix='_%s' % self._arch.name)

    def input_fn(self, path: str, repeat=None, buffer_size=None):
        """Input function for the estimator.
        Loads the train dataset from a directory given by `path` and returns input function
        that has signature () -> ({'x': features}, {'labels': labels)."""
        dataset = Dataset.from_directory(path)
        buffer_size = buffer_size or len(dataset.output_shapes[0])

        dataset = dataset.repeat(repeat).shuffle(buffer_size=buffer_size).batch(self._arch.batch_size)

        iterator = dataset.make_one_shot_iterator()
        # Use the graph invoked by estimator._train_model
        with tf.variable_scope('input_layer', reuse=True):
            features, labels = iterator.get_next()

        return {'x': features}, labels

    def predict_input_fn(self, path: str, mode='RGB'):

        if os.path.isdir(path):
            dataset = Dataset.from_directory(path)

            dataset = dataset.repeat(1).batch(self._arch.batch_size)

            iterator = dataset.make_one_shot_iterator()
            # Use the graph invoked by estimator._train_model
            with tf.variable_scope('predict_input'):
                features, _ = iterator.get_next()

        else:
            img_arr = np.asarray(Image.open(path, 'r').convert(mode))
            img_tensor = tf.convert_to_tensor(img_arr, dtype=tf.float32)
            img_tensor = tf.expand_dims(img_tensor, 0)

            dataset = tf.data.Dataset.from_tensor_slices(img_tensor)

            iterator = dataset.batch(self._arch.batch_size).make_one_shot_iterator()
            features = iterator.get_next()

        print(features)

        return {'x': features}

    def get_estimator(self):
        """Returns the estimator with the model function.
        """

        return tf.estimator.Estimator(
            model_fn=self._model_fn,
            model_dir=self._arch_cache_dir,
            params=self._params
        )

    def _model_fn(self, features, labels, mode, params=None) -> tf.estimator.EstimatorSpec:
        """Function used to be passed to an estimator and called upon train, eval or prediction.
        :returns: EstimatorSpec, a custom estimator specifications
        """

        model = Model.from_architecture(
            inputs=features,
            labels=labels,
            arch=self._arch,
            params=params,
        )

        print('Created new model {} in {}'.format(model.name, self._arch_cache_dir))

        logits = model.logits

        y_pred = tf.nn.softmax(logits)
        cls_pred = tf.argmax(y_pred)

        if mode == tf.estimator.ModeKeys.PREDICT:
            spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=cls_pred
            )

        else:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=model.labels,
                logits=logits
            )

            # reduces the cross entropy batch-tensor to a single number
            #   used for optimization of the nn
            loss = tf.reduce_mean(cross_entropy)
            # optimizer for improving nn
            optimizer = model.optimizer()

            # get the tf op for single-step optimization
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step()
            )

            # define evaluation metrics - the classification accuracy
            metrics = {
                'accuracy': tf.metrics.accuracy(labels=model.labels, predictions=y_pred)
            }

            spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                eval_metric_ops=metrics
            )

        return spec
