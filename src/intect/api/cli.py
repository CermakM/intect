#!/usr/bin/env python3
"""Main module encapsulating intect logic."""

import json
import os
import re
import sys
import time
import tensorflow as tf

import intect


FLAGS = tf.app.flags.FLAGS

# enable logging
tf.logging.set_verbosity(tf.logging.INFO)

# Define some TensorFlow app arguments
tf.app.flags.DEFINE_string(
    name='arch_dir',
    default=None,
    help="Path to directory of {.yaml, .json} files containing model specifications."
)
tf.app.flags.DEFINE_bool(
    name='train',
    default=False,
    help="Whether to train the network."
)

tf.app.flags.DEFINE_bool(
    name='eval',
    default=False,
    help="Whether to evaluate the network."
)

tf.app.flags.DEFINE_bool(
    name='predict',
    default=False,
    help="Make predictions using a pre-trained estimator."
)

tf.app.flags.DEFINE_list(
    name='images',
    default=None,
    help="Image or comma separated list of images."
)

tf.app.flags.DEFINE_bool(
    name='export',
    default=False,
    help="Save the trained estimator. The estimator will be saved in a directory specified by --export_dir"
)

tf.app.flags.DEFINE_string(
    name='export_dir',
    default='export',
    help="Specify the directory for export."
)

tf.app.flags.DEFINE_bool(
    name='json',
    default=None,
    help="Whether to use the JSON parser to parse the model architecture file."
         "By default architecture is expected to be in yaml format."
)

tf.app.flags.DEFINE_integer(
    name='train_epochs',
    default=10,
    help="Number of training epochs. This means the number of times the set is iterated over."
)


tf.app.flags.DEFINE_integer(
    name='train_steps',
    default=None,
    help="Number of training steps. This means the number of batches that is the model being trained on."
)


# noinspection PyUnusedLocal,PyUnusedLocal
def main(*args, **kwargs):  # pylint: disable=unused-argument
    """Function running main application logic."""
    if not any([FLAGS.train, FLAGS.eval, FLAGS.export, FLAGS.predict]):
        print("Either `train`, `eval`, `predict` or `export` flag must be specified.", file=sys.stderr)
        print(FLAGS.get_help())
        exit(1)

    # perform checks for predictions
    if FLAGS.predict:
        if not FLAGS.model_dir:
            print("`--model_dir` argument required", file=sys.stderr)
            print(FLAGS.get_help())
            exit(1)
        if not FLAGS.images:
            print("`--images` argument required", file=sys.stderr)
            print(FLAGS.get_help())
            exit(1)
        if any([FLAGS.train, FLAGS.eval]):
            print("`--train` or `--eval` can not be used with `--predict`", file=sys.stderr)
            print(FLAGS.get_help())
            exit(1)

    if FLAGS.arch_dir is not None:
        architectures = list()
        for root, _, walkfiles in os.walk(FLAGS.arch_dir):
            architectures.extend([os.path.join(root, f) for f in walkfiles if re.match(u"(.*)[.](y[a]?ml)$", f)])

    else:
        architectures = [FLAGS.model_arch]

    # sanity check
    assert len(architectures) > 0, "Architecture was not provided."

    # Initialize datasets
    if FLAGS.train:
        train_dataset = intect.dataset.Dataset.from_directory(FLAGS.train_dir)
    else:
        train_dataset = None

    if FLAGS.eval:
        test_dataset = intect.dataset.Dataset.from_directory(FLAGS.test_dir)
    else:
        test_dataset = None

    # Iterate over the architectures and train multiple models
    for arch_file in architectures:

        if FLAGS.json:
            arch = intect.architecture.ModelArchitecture.from_json(fp=arch_file)
        else:
            arch = intect.architecture.ModelArchitecture.from_yaml(fp=arch_file)

        estimator = intect.estimator.Estimator(
            model_architecture=arch,
            train_data=train_dataset,
            test_data=test_dataset,
            model_dir=FLAGS.model_dir or intect.utils.make_hparam_string(arch)
        )

        if FLAGS.predict:
            images = [
                intect.utils.preprocess_image(image_file=img_file)
                for img_file in FLAGS.images
            ]

            predictions = list(estimator.predict(images=images))

            print(predictions)

        # Training
        if FLAGS.train:
            start = time.time()
            tf.logging.info('Training the architecture: `%s`' % arch.name)

            estimator.train(steps=FLAGS.train_steps, num_epochs=FLAGS.train_epochs)

            tf.logging.info("Training took: {time} s.".format(time=time.time() - start))

            estimator.create_embeddings(embedding_data=train_dataset)

        # Evaluation
        if FLAGS.eval:
            start = time.time()
            tf.logging.info('Evaluating the architecture: `%s`' % arch.name)

            evaluation = estimator.evaluate()

            print('Model evaluation after %d epochs: %s' % (FLAGS.train_epochs, evaluation))
            tf.logging.info("Evaluation took: {time} s.".format(time=time.time() - start))

        # Export
        if FLAGS.export:
            export_dir = FLAGS.export_dir
            os.makedirs(export_dir, exist_ok=True)
            # export saves the model checkpoints to the export_dir
            exported = estimator.export(export_dir)
            print('Model has been exported to `%s`' % exported)


if __name__ == '__main__':

    tf.app.run(main=main)
