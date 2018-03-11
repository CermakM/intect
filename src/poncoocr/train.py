"""Main module encapsulating poncoocr logic."""

import sys
import tensorflow as tf

from src import poncoocr as pcr

FLAGS = tf.app.flags.FLAGS


def main(*args, **kwargs):  # pylint: disable=unused-argument
    """Function running main application logic."""

    if FLAGS.use_arch_dir:
        # TODO: traverse arch_dir
        architectures = list(...)
        pass
    else:
        architectures = [FLAGS.model_arch]

    # Iterate over the architectures and train multiple models
    for arch_file in architectures:
        # reset the graph so that each model has a clean initialization
        tf.reset_default_graph()

        # prefix the path with the architecture_dir
        if FLAGS.json:
            arch = pcr.architecture.ModelArchitecture.from_json(fp=arch_file)
        else:
            arch = pcr.architecture.ModelArchitecture.from_yaml(fp=arch_file)

        estim_initializer = pcr.estimator.EstimatorInitializer(model_architecture=arch)
        estimator = estim_initializer.get_estimator()

        # ---- FOR MNIST ONLY: DEBUG PURPOSES ----

        from tensorflow.examples.tutorials.mnist import input_data
        data = input_data.read_data_sets(train_dir=FLAGS.train_dir)

        # training
        train_data = data.test

        estimator.train(
            input_fn=lambda: estim_initializer.input_fn(
                features=train_data.images.reshape(-1, 28, 28, 1),
                labels=train_data.labels,
                one_hot=True,
                depth=10
            ),
            steps=FLAGS.train_steps
        )

        # validation
        validation_data = data.validation

        evaluations = estimator.evaluate(
            input_fn=lambda: estim_initializer.input_fn(
                features=validation_data.images.reshape(-1, 28, 28, 1),
                labels=validation_data.labels,
                one_hot=True,
                depth=10,
            ),
            steps=100
        )

        print('Model evaluation: ', evaluations)


if __name__ == '__main__':
    from tests import config

    tf.app.run(main=main,
               argv=[
                   sys.argv[0],
                   '--train_dir', 'src/data/mnist_data/',
                   '--model_arch', 'src/data/model/mnist-architecture.yaml',
                   '--model_dir', 'charset-arch',
                   '--buffer_size', '5000',
               ])  # FIXME: Debug
