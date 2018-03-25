"""Module containing session hooks."""

import os

import numpy as np
import tensorflow as tf

from collections import namedtuple
from tensorflow.contrib.tensorboard.plugins import projector

from . import utils

EmbeddingInput = namedtuple(typename='EmbeddingInput', field_names=['x', 'labels'])


class EmbeddingHook(tf.train.SessionRunHook):
    """Hook to extend calls to MonitoredSession.run()."""

    # noinspection SpellCheckingInspection
    def __init__(self,
                 tensors: list,
                 batch_size=None,
                 class_dct=None,
                 logdir=None,
                 every_n_steps=100):
        """Create embedding tensors.

        :param tensors: tuple or list containing feature and label tensors.
        :param class_dct: dict or None, if specified, labels will be used as keys to this dict and values outputted
        as metadata.
        """
        # initialize SessionRunHook base class at the beginning
        super(EmbeddingHook, self).__init__()

        if not (isinstance(tensors, tuple) or isinstance(tensors, list)):
            raise TypeError('`tensors` argument expected to be of type Union[tuple, list], given: {}'
                            .format(type(tensors)))

        if len(tensors) != 2:
            raise ValueError('`tensors` argument expected to be of length 2, given: {}'.format(len(tensors)))
        
        self._batch_size = batch_size

        self._class_dct = class_dct
        self._config = projector.ProjectorConfig()
        self._metadata = os.path.join(os.path.abspath(logdir), 'metadata.tsv')

        self._embedding_config = self._config.embeddings.add()
        self._embedding_config.metadata_path = self._metadata

        # Set up embedding parameters
        self._tensors = EmbeddingInput(*tensors)

        self._logdir = logdir

        # Step counter to allow evaluation only every `n` steps
        # this counter triggers every steps given by `every_n_steps`
        self._timer = tf.train.SecondOrStepTimer(every_steps=every_n_steps)
        self._iter_count = 0
        self._should_trigger = False

        # store calculated embeddings in a list
        self._embeddings = list()

        # Prototyped variables
        self._assign_op = None
        self._embedding_var = None
        self._embedding_input = None
        self._embedding_labels = None
        self._saver = None
        self._writer = None

    @property
    def embeddings(self):
        return self._embeddings

    def begin(self):
        """Called once before using the session to set up embedding variable
        and an assignement op and add them to the current graph.

        When called, the default graph is the one that will be launched in the
        session.  The hook can modify the graph by adding new operations to it.
        After the `begin()` call the graph will be finalized and the other callbacks
        can not modify the graph anymore. Second call of `begin()` on the same
        graph, should not change the graph.
        """
        self._timer.reset()
        self._iter_count = 0

        # retrieve the original tensor by its name from the graph
        self._embedding_input = utils.as_graph_element(self._tensors.x)
        self._embedding_labels = utils.as_graph_element(self._tensors.labels)

        # infer batch_size
        if self._batch_size is None:
            # try to estimate it from flags
            if tf.flags.FLAGS.is_parsed():
                self._batch_size = tf.flags.FLAGS.batch_size
            else:
                # get the default value
                self._batch_size = tf.flags.FLAGS.flag_values_dict().get('batch_size', None)

            if self._batch_size is None:
                raise ValueError("`batch_size` could not be estimated, got %s" % type(self._batch_size))

        # noinspection PyUnusedLocal
        with tf.get_default_graph().as_default() as g:  # pylint: disable=unused-variable:
            self._embedding_var = tf.Variable(tf.zeros(shape=(self._batch_size, self._embedding_input.shape[-1])),
                                              name='embedding')
            self._assign_op = self._embedding_var.assign(self._embedding_input)

        self._saver = tf.train.Saver([self._embedding_var])

    def after_create_session(self, session, coord):  # pylint: disable=unused-argument
        """Called when new TensorFlow session is created.

        This is called to signal the hooks that a new session has been created. This
        has two essential differences with the situation in which `begin` is called:

        * When this is called, the graph is finalized and ops can no longer be added
            to the graph.
        * This method will also be called as a result of recovering a wrapped
            session, not only at the beginning of the overall session.

        Args:
          session: A TensorFlow Session that has been created.
          coord: A Coordinator object which keeps track of all threads.
        """
        pass

    def before_run(self, run_context):  # pylint: disable=unused-argument
        """Called before each call to run().
        Ensures that the session fetches the assignment op and feeds
        it with images and labels data given to the `__init__()` function.

        You can return from this call a `SessionRunArgs` object indicating ops or
        tensors to add to the upcoming `run()` call.  These ops/tensors will be run
        together with the ops/tensors originally passed to the original run() call.
        The run args you return can also contain feeds to be added to the run()
        call.

        The `run_context` argument is a `SessionRunContext` that provides
        information about the upcoming `run()` call: the originally requested
        op/tensors, the TensorFlow Session.

        At this point graph is finalized and you can not add ops.

        Args:
          run_context: A `SessionRunContext` object.

        returns: `SessionRunArgs` object.
        """
        self._should_trigger = self._timer.should_trigger_for_step(self._iter_count)

        run_args = None
        if self._should_trigger:

            run_args = tf.train.SessionRunArgs(
                fetches=[self._assign_op, self._embedding_labels],
            )
            pass

        return run_args

    def after_run(self,
                  run_context,
                  run_values):
        """Called after each call to run().
        Increments the locally stored steps for the later usage with timer.

        The `run_values` argument contains results of requested ops/tensors by
        `before_run()`. Those results are saved by the saver.

        The `run_context` argument is the same one send to `before_run` call.
        `run_context.request_stop()` can be called to stop the iteration.

        If `session.run()` raises any exceptions then `after_run()` is not called.

        Args:
          run_context: A `SessionRunContext` object.
          run_values: A SessionRunValues object.
        """
        if self._should_trigger:
            # add new embedddings
            results, labels = run_values.results

            self._embeddings.extend(results)

            labels = np.argmax(labels, axis=1)
            if self._class_dct:
                labels = [chr(int(self._class_dct[label])) for label in labels]

            # write metadata
            with open(self._metadata, 'a') as meta:
                for i, label in enumerate(labels):
                    meta.write("{!s}\n".format(label))

            # Save the embedding
            tf.logging.info('Saving checkpoint for embeddings for global step: %d' % self._iter_count)
            self._saver.save(sess=run_context.session, save_path=os.path.join(self._logdir, 'embedding.ckpt'),
                             global_step=tf.train.get_global_step())

            # give timer the information that it has been triggered
            _ = self._timer.update_last_triggered_step(self._iter_count)

        self._iter_count += 1

    def end(self, session):  # pylint: disable=unused-argument
        """Called at the end of session.

        The `session` argument can be used in case the hook wants to run final ops,
        such as saving a last checkpoint.

        If `session.run()` raises exception other than OutOfRangeError or
        StopIteration then `end()` is not called.
        Note the difference between `end()` and `after_run()` behavior when
        `session.run()` raises OutOfRangeError or StopIteration. In that case
        `end()` is called but `after_run()` is not called.

        Args:
          session: A TensorFlow Session that will be soon closed.
        """

        # instantiate custom saver writer
        self._writer = tf.summary.FileWriter(self._logdir, graph=tf.get_default_graph())

        # add tensor to the embedding config
        self._embedding_config.tensor_name = self._embedding_var.name

        # Store the config file used by the embedding projector
        projector.visualize_embeddings(self._writer, self._config)

        self._saver.save(sess=session, save_path=os.path.join(self._logdir, 'embedding.ckpt'),
                         global_step=self._iter_count)


