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
from .session import EmbeddingHook, HookModes


class Estimator(object):

    __model_cache_dir = '.model_cache/'

    def __init__(self,
                 model_architecture: ModelArchitecture,
                 train_data: Dataset=None,
                 test_data: Dataset=None,
                 model_fn=None,
                 model_dir=None,
                 params: dict = None):
        """Initialize estimator parameters and create training and evaluation hooks."""
        self._arch = model_architecture
        self._train_data = train_data
        self._test_data = test_data
        self._params = params or dict()

        self._chief_hooks = list()

        # create cache directory if necessary
        if not os.path.isdir(self.__model_cache_dir):
            os.mkdir(self.__model_cache_dir)

        # create model specific sub directory in cache dir
        self._cache_dir = model_dir
        if self._cache_dir is None:
            self._cache_dir = tempfile.mkdtemp(dir=os.path.join(self.__model_cache_dir),
                                               prefix='arch_',
                                               suffix='_%s' % self._arch.name)
        else:
            self._cache_dir = os.path.join(self.__model_cache_dir, self._cache_dir)
            if not os.path.isdir(self._cache_dir):
                os.mkdir(path=self._cache_dir)

        self._embedding_dir = os.path.join(self._cache_dir, 'projector')
        if not os.path.isdir(self._embedding_dir):
            os.mkdir(self._embedding_dir)

        if tf.flags.FLAGS.is_parsed():
            default_embedding_size = tf.flags.FLAGS.embedding_size
        else:
            # set an implicit value in case no other declaration found
            default_embedding_size = min(2048, len(train_data))

        self._embedding_size = min(self._params.get('embedding_size', default_embedding_size), len(train_data))
        # self._sprite, self._metadata = utils.make_sprite_image(
        #     images=train_data.features[:self._embedding_size],
        #     # convert labels to classes and use it as metadata
        #     metadata=utils.label_to_class(
        #         labels=train_data.labels[:self._embedding_size],
        #         class_dct=train_data.classes,
        #         decode=True,
        #     ),
        #     num_images=self._embedding_size,
        #     dir_path=self._embedding_dir,
        #     renormalize=True,
        # )

        # create logging hook for training accuracy and learning rate
        logging_hook = tf.train.LoggingTensorHook(
            tensors={
                'accuracy': 'log_accuracy',
                'loss': 'log_loss',
                'learning_rate': 'log_learning_rate',
            },
            every_n_iter=100
        )

        # create embedding hook
        self._embedding_hook = EmbeddingHook(
            tensors=['embedding_input', 'embedding_labels'],
            embedding_size=self._embedding_size,
            class_dct=self._train_data.classes,
            logdir=self._embedding_dir,
        )

        self._chief_hooks.extend([logging_hook])

        self._estimator = self._get_estimator(model_fn=model_fn)

    @property
    def test_data(self):
        return self._test_data

    @property
    def train_data(self):
        return self._train_data

    @property
    def classes(self):
        if not any([self._train_data, self._test_data]):
            raise AttributeError("No data available to the Estimator, "
                                 "property `classes` cannot be accessed.")
        return (self._train_data or self._test_data).classes

    def train(self, train_data: Dataset = None, steps=None, num_epochs=1, buffer_size=None, hooks=None):
        """Train the estimator.

        :param steps: number of steps to be the estimator trained for.
        :param buffer_size: size of the buffer that is used for shuffling.
        If `None`, buffer size is chosen according to the size of the training data.
        """
        if not train_data:
            train_data = self._train_data

        assert train_data is not None, "`train_data` has not been provided."

        train_hooks = self._chief_hooks.copy()
        train_hooks.append(self._embedding_hook)
        train_hooks.extend(hooks or [])

        if steps is None:
            if len(train_data) < self._arch.batch_size:
                steps = len(train_data)
            else:
                # number of steps in one epoch
                steps = len(train_data) // self._arch.batch_size

        self._estimator.train(
            input_fn=lambda: self._dataset_input_fn(
                dataset=train_data,
                batch_size=self._arch.batch_size,
                repeat=None,
                buffer_size=buffer_size or len(train_data)
            ),
            steps=steps * num_epochs,
            hooks=train_hooks,
        )

    def create_embeddings(self, embedding_data: Dataset = None):
        """Train the embeddings for visualizer."""
        if not embedding_data:
            embedding_data = self._train_data

        assert embedding_data is not None, "`embedding_data` has not been provided."

        # allow embedding hook to run
        self._embedding_hook.mode = HookModes.RUN

        _ = self._estimator.evaluate(
            input_fn=lambda: self._dataset_input_fn(
                dataset=embedding_data,
                batch_size=self._embedding_size,
                repeat=None,
                buffer_size=self._embedding_size,
            ),
            steps=1,
            hooks=[self._embedding_hook],
        )

        # place the embedding hook to the init mode again
        self._embedding_hook.mode = HookModes.INIT_ONLY

    def evaluate(self, steps=None, hooks=None) -> dict:
        """Evaluate accuracy of the estimator."""

        eval_hooks = self._chief_hooks.copy()
        eval_hooks.extend(hooks or [])

        if not self.test_data:
            raise TypeError("`test_data` has not been provided for evaluation.")
        results = self._estimator.evaluate(
            input_fn=lambda: self._dataset_input_fn(
                dataset=self._test_data,
                batch_size=self._arch.batch_size,
                repeat=1 if steps is None else None,
                buffer_size=len(self._test_data) // 10,  # shuffle only 10% of the test data
            ),
            steps=steps,
            hooks=eval_hooks,
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

    def save(self):
        """Saves the estimator for future restore and serving."""
        raise NotImplementedError

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

    def _dataset_input_fn(self, dataset: Dataset, repeat=None, shuffle=True, batch_size=None, buffer_size=None):
        """Input function for the estimator.
        Loads the dataset from a directory given by `path` and returns tuple ({'x': features}, {'labels': labels)."""
        buffer_size = buffer_size or len(dataset)

        tf_dataset = tf.data.Dataset.from_tensor_slices((dict(x=dataset.features), dataset.labels))
        tf_dataset = tf_dataset.repeat(repeat)
        if shuffle:
            tf_dataset = tf_dataset.shuffle(buffer_size=buffer_size or len(dataset))

        if batch_size:
            tf_dataset = tf_dataset.batch(batch_size=batch_size)

        return tf_dataset

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

    def _get_estimator(self, model_fn=None, model_dir=None, save_summaries=True):
        """Returns the estimator with the model function.
        :param model_fn: custom model function which will be passed to the Estimator. See `tf.estimator.Estimator`
        documentation for more info.
        """

        return tf.estimator.Estimator(
            model_fn=model_fn or self._model_fn,
            model_dir=model_dir or self._cache_dir,
            params=self._params,
            config=tf.estimator.RunConfig(
                # Passing None to save_summary_steps disables Estimator's SummaryHook
                save_summary_steps=100 if save_summaries else None,
            )
        )

    def _model_fn(self, features, labels, mode, params=None) -> tf.estimator.EstimatorSpec:
        """Function used to be passed to an estimator and called upon train, eval or prediction.
        :returns: EstimatorSpec, a custom estimator specifications
        """

        tf.summary.image('images', features['x'], max_outputs=3)

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
            tf.summary.scalar('train_loss', loss)

            # add to the graph an embedding tensor which will be fetched by EmbeddingHook
            tf.identity(model.hidden_layers[-1], 'embedding_input')
            tf.identity(model.labels, 'embedding_labels')

            # initialize optimizer
            optimizer = model.optimizer(learning_rate=model.learning_rate)
            tf.logging.info('Optimizer: %s' % optimizer)

            # get the tf op for single-step optimization
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step()
            )

            # create confusion matrix
            num_classes = len(self.classes)
            batch_confusion = tf.confusion_matrix(
                labels=tf.argmax(model.labels, axis=1),
                predictions=y_pred_cls,
                num_classes=num_classes
            )
            confusion = tf.Variable(
                initial_value=tf.zeros(shape=(num_classes, num_classes), dtype=tf.int32),
                name='confusion',
            )
            confusion = confusion.assign(confusion + batch_confusion)
            confusion_image = tf.reshape(
                tf.cast(confusion, tf.float32),
                [1, num_classes, num_classes, 1]
            )

            tf.summary.image(tensor=confusion_image, name='confusion')

            # define evaluation metrics - the classification accuracy
            # NOTE: assuming model labels to be one-hot encoded, therefore argmax needs to be performed first
            accuracy = tf.metrics.accuracy(labels=tf.argmax(model.labels, axis=1), predictions=y_pred_cls,
                                           name='eval_accuracy')

            tf.summary.scalar(tensor=accuracy[1], name='train_accuracy')

            metrics = {
                'accuracy': accuracy
            }

            # tensors to be logged
            tf.identity(accuracy[1], 'log_accuracy')
            tf.identity(loss, 'log_loss')
            tf.identity(model.learning_rate, 'log_learning_rate')

            spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                eval_metric_ops=metrics,
                scaffold=tf.train.Scaffold(summary_op=tf.summary.merge_all())
            )

        return spec
