"""Module containing common variables and configurations for poncoocr modules."""

import os
import tensorflow as tf

from src import data

_data_dir = os.path.dirname(data.__file__)


# Define TensorFlow app arguments

tf.app.flags.DEFINE_bool(
    name='json',
    default=False,
    help="Whether to use the JSON parser to parse the model architecture file."
         "By default architecture is expected to be in yaml format."
)

tf.app.flags.DEFINE_string(
    name='use_arch_dir',
    default=None,
    help="Path to directory of {.yaml, .json} files containing model specifications."
)

# Define TensorFlow numeric flags

tf.app.flags.DEFINE_integer(
    name='buffer_size',
    default=20000,
    help="Size of the buffer which is used for shuffling images."
         "Indicates the number of images that can be shuffled."
)

tf.app.flags.DEFINE_integer(
    name='train_epochs',
    default=10,
    help="Number of training steps. This means the number of batches that is the model being trained on."
)

tf.app.flags.DEFINE_integer(
    name='test_steps',
    default=500,
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
    help="Path to the directory storing sprites."
)
