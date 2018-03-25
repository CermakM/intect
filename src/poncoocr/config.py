"""Module containing common variables and configurations for poncoocr modules."""

import os
import tensorflow as tf

from src import data

_data_dir = os.path.dirname(data.__file__)


# Define TensorFlow app arguments

tf.app.flags.DEFINE_bool(
    name='train',
    default=True,
    help="Whether to train the network."
)

tf.app.flags.DEFINE_bool(
    name='evaluate',
    default=True,
    help="Whether to evaluate the network."
)

tf.app.flags.DEFINE_bool(
    name='json',
    default=None,
    help="Whether to use the JSON parser to parse the model architecture file."
         "By default architecture is expected to be in yaml format."
)

tf.app.flags.DEFINE_bool(
    name='save',
    default=None,
    help="Save the trained estimator. The estimator will be saved in a directory called `export/$ARCH_NAME`"
)

tf.app.flags.DEFINE_string(
    name='arch_dir',
    default=None,
    help="Path to directory of {.yaml, .json} files containing model specifications."
)


# Define TensorFlow numeric flags

tf.app.flags.DEFINE_integer(
    name='train_epochs',
    default=10,
    help="Number of training epochs. This means the number of times the set is iterated over."
)

tf.app.flags.DEFINE_integer(
    name='batch_size',
    default=None,
    help="Batch size which will be used for training"
)

tf.app.flags.DEFINE_float(
    name='learning_rate',
    default=None,
    help="Learning rate parameter used for training."
)

tf.app.flags.DEFINE_integer(
    name='embedding_size',
    default=2048,
    help="Number of images passed to the embedding."
)

tf.app.flags.DEFINE_integer(
    name='train_steps',
    default=None,
    help="Number of training steps. This means the number of batches that is the model being trained on."
)

# Define TensorFlow string flags

tf.app.flags.DEFINE_string(
    name='test_dir',
    default=os.path.join(_data_dir, 'char-dataset/test_data'),
    help="Path to the directory storing test data."
)

tf.app.flags.DEFINE_string(
    name='train_dir',
    default=os.path.join(_data_dir, 'char-dataset/train_data'),
    help="Path to the directory storing train data."
)

tf.app.flags.DEFINE_string(
    name='model_arch',
    default=os.path.join(_data_dir, 'model/default-architecture.yaml'),
    help="Path to the directory storing model architecture."
)

tf.app.flags.DEFINE_string(
    name='sprite_dir',
    default=os.path.join(_data_dir, 'sprites/'),
    help="Path to the directory storing data sprites and their metadata."
)
