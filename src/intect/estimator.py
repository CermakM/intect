"""Module containing estimator for intect engine."""

import os
import tempfile
import typing

import numpy as np
import tensorflow as tf

from intect import config
from intect.architecture import ModelArchitecture
from intect.dataset import Dataset
from intect.model import Model
from intect.session import EmbeddingHook, HookModes

from tensorflow.core.framework.tensor_pb2 import TensorProto
from tensorflow.contrib.lookup import index_to_string_table_from_tensor


DEFAULT_SERVING_SIGNATURE_DEF_KEY = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY


class Estimator(object):
    """Estimator class."""

    __model_cache_dir = '.model_cache/'

    def __init__(self,
                 model_architecture: ModelArchitecture,
                 train_data: Dataset=None,
                 test_data: Dataset=None,
                 model_fn=None,
                 model_dir=None,
                 params: dict = None):

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
            default_embedding_size = config.EMBEDDING_SIZE

        self._embedding_size = self._params.get('embedding_size', default_embedding_size)

        # create logging hook for training accuracy and learning rate
        logging_hook = tf.train.LoggingTensorHook(
            tensors={
                'accuracy': 'log_accuracy',
                'loss': 'log_loss',
                'learning_rate': 'log_learning_rate',
            },
            every_n_iter=100
        )

        # get the classes
        if any([train_data is not None, test_data is not None]):
            classes = getattr(self._train_data, 'classes', None) or self._test_data.classes

            # transform the classes to ascii repr
            classes = {k: chr(int(v)) for (k, v) in classes.items()}
        else:
            classes = None
        self._classes = classes

        # create embedding hook
        self._embedding_hook = EmbeddingHook(
            tensors=[
                config.EMBEDDING_TENSORS.BATCH_FEATURES,
                config.EMBEDDING_TENSORS.BATCH_LABELS,
                config.EMBEDDING_TENSORS.EMBEDDING_INPUT,
            ],
            embedding_size=self._embedding_size,
            class_dct=classes,
            logdir=self._embedding_dir,
        )

        self._chief_hooks.extend([logging_hook])

        self._estimator = self._get_estimator(model_fn=model_fn)

    @property
    def model_dir(self):
        return self._cache_dir

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
        return self._classes

    def train(self, train_data: Dataset = None, steps=None, num_epochs=1, buffer_size=None, hooks=None):
        """Train the estimator.

        :param train_data: Dataset, Dataset to be used

            Dataset class or object implementing `features` and `labels` properties
            Will be fed into the input_fn.

            By default the method uses the dataset provided as `train_data` argument when
            initializing the estimator. If specified, the argument of the `train` function
            is privileged.

        :param steps: int, number of steps to be the estimator trained for.
        :param num_epochs: int, number of epochs to be used for training

            As an epoch is by default understood a single iteration over the whole dataset.
            If provided along with `steps` argument, the epoch will be understood as the
            iteration over the number of steps regardless size of the dataset.

        :param buffer_size: int, size of the buffer that is used for shuffling.

            If `None`, buffer size is chosen according to the size of the training data.

        :param hooks: list, training hooks

            Hooks to be ran in MonitoredSession. Can implement fe. `begin` and `after_run`
            methods which will be called by the session.
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

    def evaluate(self, test_data: Dataset = None, steps=None, hooks=None) -> dict:
        """Evaluate accuracy of the estimator."""

        eval_hooks = self._chief_hooks.copy()
        eval_hooks.extend(hooks or [])

        if not test_data:
            test_data = self._test_data

        assert test_data is not None, "`test_data` has not been provided for evaluation."

        results = self._estimator.evaluate(
            input_fn=lambda: self._dataset_input_fn(
                dataset=test_data,
                batch_size=self._arch.batch_size,
                repeat=1 if steps is None else None,
                buffer_size=len(test_data) // 10,  # shuffle only 10% of the test data
            ),
            steps=steps,
            hooks=eval_hooks,
        )

        return results

    def predict(self, images: list) -> typing.Generator:
        """Produces class predictions about list of images given by `images`.

        :param images: list containing images to predict

        :returns: generator yielding per-image predictions
        """
        if not isinstance(images, list):
            raise TypeError("Expected type `list`, got `{}`".format(type(images)))

        predictions = self._estimator.predict(
            input_fn=lambda: self._predict_input_fn(images=images)
        )

        return predictions

    def export(self, export_dir: str):
        """Exports the estimator for future restore and serving.

        :returns: path to the exported directory
        """
        return self._estimator.export_savedmodel(
            export_dir_base=export_dir,
            serving_input_receiver_fn=tf.estimator.export.build_raw_serving_input_receiver_fn(
                features={
                    'x': tf.placeholder(shape=(None, *config.IMAGE_SHAPE, 1),
                                        dtype=tf.float32)
                }
            )
        )

    @staticmethod
    def _serving_input_fn():
        """Input receiver function for serving."""
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

    def _dataset_input_fn(self,
                          dataset: Dataset,
                          repeat=None,
                          shuffle=True,
                          batch_size=None,
                          buffer_size=None) -> tf.data.Dataset:
        """Input function for the estimator.
        Loads the dataset from a directory given by `path` and returns tuple ({'x': features}, {'labels': labels)."""
        buffer_size = buffer_size or len(dataset)

        tf_dataset = tf.data.Dataset.from_tensor_slices((dict(x=dataset.features), dataset.labels))
        tf_dataset = tf_dataset.repeat(repeat)
        if shuffle:
            tf_dataset = tf_dataset.shuffle(buffer_size=buffer_size or len(dataset))

        if batch_size:
            tf_dataset = tf_dataset.batch(batch_size=batch_size or self._arch.batch_size)

        return tf_dataset

    def _predict_input_fn(self, images: typing.Iterable) -> tf.data.Dataset:
        """Make predictions about each image from `images` list."""

        if not isinstance(images, typing.Iterable):
            raise TypeError("Argument `fp` expected to be of type `{}`, got `{}`"
                            .format(typing.Iterable, type(images)))

        # convert to tensor
        img_tensor = tf.convert_to_tensor(np.stack(images), dtype=tf.float32)

        # create dataset
        tf_dataset = tf.data.Dataset.from_tensor_slices(dict(x=img_tensor))

        return tf_dataset.batch(self._arch.batch_size)

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
        y_pred_score, y_pred_cls = tf.nn.top_k(y_pred)

        default_export_output = tf.estimator.export.ClassificationOutput(y_pred)

        if mode == tf.estimator.ModeKeys.PREDICT:
            # restore class array from serialized tensor proto
            label_tensor = self._tensor_from_proto(os.path.join(self._cache_dir, config.LABEL_TENSOR_PROTO_FP))

            # create lookup table
            lookup_cls_table = index_to_string_table_from_tensor(label_tensor)

            # take all of the classes
            pred_scores, pred_classes = tf.nn.top_k(y_pred, k=len(label_tensor))

            # cast to the matching dtype for lookup
            pred_classes = tf.cast(pred_classes, dtype=tf.int64)

            cls = lookup_cls_table.lookup(tf.to_int64(y_pred_cls))
            classes = lookup_cls_table.lookup(tf.to_int64(pred_classes))

            spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={
                    'cls': cls,
                    'score': y_pred_score,
                },
                export_outputs={
                    'prediction': tf.estimator.export.PredictOutput(cls),
                    'confidence': tf.estimator.export.PredictOutput(tf.reduce_max(y_pred, axis=1)),
                    'classes': tf.estimator.export.PredictOutput(classes),
                    'scores': tf.estimator.export.PredictOutput(pred_scores),
                    # include the default signature def as well
                    DEFAULT_SERVING_SIGNATURE_DEF_KEY: default_export_output
                }
            )

        else:
            num_classes = len(self.classes)

            # export class metadata for prediction mode
            label_data = np.array([v for k, v in sorted(self.classes.items())], dtype=np.unicode_)
            self._export_tensor_proto(proto_path=config.LABEL_TENSOR_PROTO_FP, tensor_data=label_data)

            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=model.labels,
                logits=y_pred
            )

            # reduces the cross entropy batch-tensor to a single number
            # used for optimization of the nn
            loss = tf.reduce_mean(cross_entropy)
            tf.summary.scalar('train_loss', loss)

            # add to the graph the tensors which will be fetched by EmbeddingHook
            tf.identity(model.x, config.EMBEDDING_TENSORS.BATCH_FEATURES)
            tf.identity(model.labels, config.EMBEDDING_TENSORS.BATCH_LABELS)
            tf.identity(model.hidden_layers[-1], config.EMBEDDING_TENSORS.EMBEDDING_INPUT)

            # initialize optimizer
            optimizer = model.optimizer(learning_rate=model.learning_rate)
            tf.logging.info('Optimizer: %s' % optimizer)

            # get the tf op for single-step optimization
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step()
            )

            # create confusion matrix
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
                export_outputs={
                    DEFAULT_SERVING_SIGNATURE_DEF_KEY: default_export_output
                }
            )

        return spec

    def _export_tensor_proto(self, proto_path, tensor_data: np.ndarray):
        """Serialize and export metadata to a file as a tensor proto.

        The exported file will be placed into the `model_dir` as <fname>.proto (by default label_meta.proto)
        """
        fp = os.path.join(self._cache_dir, proto_path)

        with open(fp, 'wb') as meta:
            # create tensor proto
            meta.write(tf.make_tensor_proto(tensor_data).SerializeToString())

        return fp

    @staticmethod
    def _tensor_from_proto(fp) -> np.ndarray:
        """Restores tensor from .proto` file.

        :param fp: str, path to  .proto file
        :returns: ndarray of tensor content
        """

        with open(fp, 'rb') as proto_file:
            serialized_tensor_proto = proto_file.read()

        # convert to TensorProto
        tensor_proto = TensorProto.FromString(serialized_tensor_proto)

        return tf.make_ndarray(tensor_proto)
