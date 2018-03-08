"""Module containing estimator for poncoocr engine."""

import tensorflow as tf

from .architecture import ModelArchitecture
from .dataset import Dataset
from .model import Model


class EstimatorInitializer(object):

    def __init__(self, model_architecture: ModelArchitecture, params: dict = None):
        """Initialize estimator initializer with a model to be used for the estimator."""

        self._arch = model_architecture
        self._params = params

    def input_fn(self, path: str, repeat=None, buffer_size=None):
        """Input function for the estimator.
        Loads the train dataset from a directory given by `path` and returns input function
        that has signature () -> ({'x': features}, {'labels': labels)."""
        dataset = Dataset.from_directory(path)
        buffer_size = buffer_size or len(dataset.output_shapes[0])

        dataset = dataset.repeat(repeat).shuffle(buffer_size=buffer_size).batch(self._arch.batch_size)

        iterator = dataset.make_one_shot_iterator()
        with tf.variable_scope('input_layer', reuse=True):
            features, labels = iterator.get_next()

        return lambda: ({'x': features}, labels)

    def model_fn(self, features, labels, mode, params=None) -> tf.estimator.EstimatorSpec:
        """Function used to be passed to an estimator and called upon train, eval or prediction.
        :returns: EstimatorSpec, a custom estimator specifications
        """

        # Pass the features and labels to the model and assign them to the new graph
        model = Model.from_architecture(
            inputs=features,
            labels=labels,
            arch=self._arch,
            params=params,
        )

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

    def get_estimator(self, model_dir=None, params=None):
        """Returns the estimator with the model function.
        :param model_dir: directory to save the model, graph and checkpoints to
        :param params: parameters to be passed to `model_fn`
        """

        params = params or dict()

        return tf.estimator.Estimator(
            model_fn=self.model_fn,
            model_dir=model_dir,
            params=params
        )
