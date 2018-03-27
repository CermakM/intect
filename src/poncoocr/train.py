"""Main module encapsulating poncoocr logic."""

import os
import re
import sys
import time
import tensorflow as tf

from src import poncoocr as pcr

# enable logging
tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS


def main(*args, **kwargs):  # pylint: disable=unused-argument
    """Function running main application logic."""
    # Initialize datasets
    train_dataset = pcr.dataset.Dataset.from_directory(FLAGS.train_dir)
    test_dataset = pcr.dataset.Dataset.from_directory(FLAGS.test_dir)

    if FLAGS.arch_dir is not None:
        architectures = list()
        for root, _, walkfiles in os.walk(FLAGS.arch_dir):
            architectures.extend([os.path.join(root, f) for f in walkfiles if re.match(u"(.*)[.](y[a]?ml)$", f)])

    else:
        architectures = [FLAGS.model_arch]

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

        if FLAGS.train:
            start = time.time()
            tf.logging.info('Training the architecture: `%s`' % arch.name)

            estimator.train(steps=FLAGS.train_steps, num_epochs=FLAGS.train_epochs)

            tf.logging.info("Training took: {time} s.".format(time=time.time() - start))

            estimator.create_embeddings(embedding_data=train_dataset)

        if FLAGS.eval:
            evaluation = estimator.evaluate()
            print('Model evaluation after %d epochs: %s' % (FLAGS.train_epochs, evaluation))

        if FLAGS.save:
            raise NotImplementedError
            # export_dir = "save/%s" % arch.name
            # os.makedirs(export_dir, exist_ok=True)
            # estimator.save()


if __name__ == '__main__':
    from tests import config

    tf.app.run(main=main, argv=[sys.argv[0],
                                '--train_steps', '1000',
                                '--train_epochs', '1',
                                # '--train_dir', config.TEST_DATASET_PATH,
                                # '--test_dir', config.TEST_DATASET_PATH,
                                # '--model_arch', config.TEST_ARCHITECTURE_YAML,
                                '--train_dir', '/home/macermak/code/thesis/ponco-ocr/src/data/num-dataset/train_data',
                                '--test_dir', '/home/macermak/code/thesis/ponco-ocr/src/data/num-dataset/test_data',
                                '--model_arch', '/home/macermak/code/thesis/ponco-ocr/src/data/num-dataset/num-architecture.yaml',
                                # '--notrain',
                                # '--noeval'
                                ])
