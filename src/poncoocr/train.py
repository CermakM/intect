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

        print('Training the model: `%s`' % arch.name)
        estimator.train(
            input_fn=lambda: estim_initializer.dir_input_fn(
                path=FLAGS.test_dir,
                buffer_size=FLAGS.buffer_size,
            ),
            steps=FLAGS.train_steps
        )

        print('Evaluating the model: `%s`' % arch.name)
        evaluations = estimator.evaluate(
            input_fn=lambda: estim_initializer.dir_input_fn(
                path=FLAGS.test_dir,
                repeat=10,
                buffer_size=FLAGS.buffer_size,
            ),
        )

        print('Model `%s` evaluation: ' % arch.name, evaluations)


if __name__ == '__main__':
    tf.app.run(main=main,
               argv=[
                   sys.argv[0],
                   '--model_dir', 'charset-arch',
                   '--buffer_size', '20000',
               ])  # FIXME: Debug
