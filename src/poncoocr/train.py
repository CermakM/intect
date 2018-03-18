"""Main module encapsulating poncoocr logic."""

import os
import re
import sys
import time
import tensorflow as tf

from src import poncoocr as pcr

FLAGS = tf.app.flags.FLAGS


def main(*args, **kwargs):  # pylint: disable=unused-argument
    """Function running main application logic."""
    # Initialize datasets
    train_dataset = pcr.dataset.Dataset.from_directory(FLAGS.test_dir)  # FIXME
    # test_dataset = pcr.dataset.Dataset.from_directory(FLAGS.test_dir)

    if FLAGS.use_arch_dir is not None:
        architectures = list()
        for root, _, walkfiles in os.walk(FLAGS.use_arch_dir):
            architectures.extend([os.path.join(root, f) for f in walkfiles if re.match(u"(.*)[.](y[a]?ml)$", f)])

    else:
        architectures = [FLAGS.model_arch]

    # Iterate over the architectures and train multiple models
    for arch_file in architectures:

        # prefix the path with the architecture_dir
        if FLAGS.json:
            arch = pcr.architecture.ModelArchitecture.from_json(fp=arch_file)
        else:
            arch = pcr.architecture.ModelArchitecture.from_yaml(fp=arch_file)

        estimator = pcr.estimator.Estimator(
            model_architecture=arch,
            train_data=train_dataset,
            test_data=None,
            model_dir=pcr.utils.make_hparam_string(arch)
        )

        start = time.time()
        print('Training the architecture: `%s`' % arch.name)
        estimator.train(num_epochs=FLAGS.train_epochs)
        print("Training took: {time} s.".format(time=time.time() - start))

        # print('Evaluating the architecture: `%s`' % arch.name)
        # evaluations = estimator.evaluate()
        # print('Model `%s` evaluation: ' % arch.name, evaluations)


if __name__ == '__main__':
    tf.app.run(main=main,
               argv=[
                   sys.argv[0],
                   # '--train_dir', config.TEST_DATASET_PATH,
                   # '--test_dir', config.TEST_DATASET_PATH,
                   '--model_arch', 'src/data/model/mango-arch.yaml',
                   # '--use_arch_dir', 'src/data/model',
                   '--buffer_size', '30000',
                   '--train_epochs', '50',
               ])  # FIXME: Debug
