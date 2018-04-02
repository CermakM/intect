"""Main module encapsulating poncoocr logic."""

import os
import re
import sys
import time
import tensorflow as tf

import poncoocr as pcr

# enable logging
tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS


# noinspection PyUnusedLocal,PyUnusedLocal
def main(*args, **kwargs):  # pylint: disable=unused-argument
    """Function running main application logic."""

    if not any([FLAGS.train, FLAGS.eval, FLAGS.export]):
        raise ValueError("Either `train`, `eval` or `export` flag must be specified.")

    if FLAGS.arch_dir is not None:
        architectures = list()
        for root, _, walkfiles in os.walk(FLAGS.arch_dir):
            architectures.extend([os.path.join(root, f) for f in walkfiles if re.match(u"(.*)[.](y[a]?ml)$", f)])

    else:
        architectures = [FLAGS.model_arch]

    # Initialize datasets
    if FLAGS.train:
        train_dataset = pcr.dataset.Dataset.from_directory(FLAGS.train_dir)
    else:
        train_dataset = None

    if FLAGS.eval:
        test_dataset = pcr.dataset.Dataset.from_directory(FLAGS.test_dir)
    else:
        test_dataset = None

    # Iterate over the architectures and train multiple models
    for arch_file in architectures:

        if FLAGS.json:
            arch = pcr.architecture.ModelArchitecture.from_json(fp=arch_file)
        else:
            arch = pcr.architecture.ModelArchitecture.from_yaml(fp=arch_file)

        estimator = pcr.estimator.Estimator(
            model_architecture=arch,
            train_data=train_dataset,
            test_data=test_dataset,
            model_dir=pcr.utils.make_hparam_string(arch)
        )

        # Training
        if FLAGS.train:
            start = time.time()
            tf.logging.info('Training the architecture: `%s`' % arch.name)

            estimator.train(steps=FLAGS.train_steps, num_epochs=FLAGS.train_epochs)

            tf.logging.info("Training took: {time} s.".format(time=time.time() - start))

            estimator.create_embeddings(embedding_data=train_dataset)

        # Evaluation
        if FLAGS.eval:
            evaluation = estimator.evaluate()
            print('Model evaluation after %d epochs: %s' % (FLAGS.train_epochs, evaluation))

        # Export
        if FLAGS.export:
            export_dir = FLAGS.export
            os.makedirs(export_dir, exist_ok=True)
            # export saves the model checkpoints to the export_dir
            estimator.export(export_dir)


if __name__ == '__main__':

    tf.app.run(main=main, argv=[
        sys.argv[0],
    ])
