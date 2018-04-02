"""Module containing common utility functions."""

import os
import time
import threading

import numpy as np
import tensorflow as tf

from collections import Counter
from collections.abc import Mapping
from itertools import cycle
from math import sqrt, ceil
from matplotlib.pyplot import plot as plt
from PIL import Image

import poncoocr as pcr


# FUNCTIONS

def as_graph_element(obj):
    """Retrieves Graph element."""
    graph = tf.get_default_graph()
    if not isinstance(obj, str):
        if not hasattr(obj, "graph") or obj.graph != graph:
            raise ValueError("Passed %s should have graph attribute that is equal "
                             "to current graph %s." % (obj, graph))
        return obj
    if ":" in obj:
        element = graph.as_graph_element(obj)
    else:
        element = graph.as_graph_element(obj + ":0")
        # Check that there is no :1 (e.g. it's single output).
        try:
            graph.as_graph_element(obj + ":1")
        except (KeyError, ValueError):
            pass
        else:
            raise ValueError("Name %s is ambiguous, "
                             "as this `Operation` has multiple outputs "
                             "(at least 2)." % obj)
    return element


def label_to_class(labels, class_dct, one_hot=True, decode=False):
    """Convert a tensor of labels into a tensor of classes given a class dict.

    :param labels: np.array or iterable of labels
    :param class_dct: dict, mapping from label index to a real class
    :param one_hot: bool, whether `labels` is one_hot encoded
    :param decode: whether to decode label strings from unicode representation

    :returns: tensor of the shape (len_labels,) of classed labels
    """
    # TODO: handle TensorFlow tensors as well
    if one_hot:
        labels = np.argmax(labels, axis=1)

    # labels are unicode representation of a letter, if `decode` is specified, return
    # the decoded character
    if decode:
        classes = np.array([chr(int(class_dct[label])) for label in labels], dtype=np.unicode_)
    else:
        classes = np.array([class_dct[label] for label in labels], dtype=np.unicode_)

    return classes


def make_hparam_string(arch: "pcr.architecture.ModelArchitecture") -> str:
    """Make a hyper parameter string from architecture spec.
    The string can then be used to distinguish various runs in tensorboard.
    """
    types = [layer.type for layer in arch.layers]
    bag = Counter(types)

    hparam_string = "{name},lr={lr},bs={bs},conv={conv},fcl={fcl}".format(
        name=arch.name,
        lr=arch.learning_rate,
        bs=arch.batch_size,
        conv=bag.get('conv2d', 0),
        fcl=bag.get('dense', 1) - 1
    )

    return hparam_string


def make_sprite_image(images,
                      metadata,
                      num_images=1024,
                      thumbnail=pcr.config.THUMBNAIL_SHAPE,
                      fill='#fff',
                      renormalize=True,
                      dir_path=None) -> tuple:
    """Create sprite image from a set of images.

    :returns: "tuple"(str, str), absolute path to the newly created sprite and metadata
    """

    assert num_images > 0, "Number of images must be > 0, given: %d" % num_images

    if renormalize:
        images *= 255

    # turn the images into uint8 format - further improves renormalization
    images = np.uint8(images)

    iterator = cycle(zip(images, metadata))
    "iterator yielding tuple of (image, metadata)"

    sprite_fp = os.path.join(os.path.abspath(dir_path or ''), 'sprite.png')
    metadata_fp = os.path.join(os.path.abspath(dir_path or ''), 'metadata.tsv')

    if os.path.isdir(sprite_fp):
        tf.logging.info("Sprite `%s` already exists in the directory, skipping." % sprite_fp)
        return sprite_fp, metadata_fp

    # open file handle for metadata
    meta_tsv = open(metadata_fp, 'w')

    # create white board
    board = Image.new(mode='L', size=tuple(int(ceil(sqrt(num_images))) * np.array(thumbnail)), color=fill)

    pos = 0, 0
    for im_index in range(1, num_images + 1):
        img, meta = next(iterator)
        img = img.reshape(thumbnail)
        img = Image.fromarray(img).convert('L')
        board.paste(img, pos)

        if im_index % ceil(sqrt(num_images)) == 0:
            pos = 0, pos[1] + thumbnail[1]
        else:
            pos = pos[0] + thumbnail[0], pos[1]

        meta_tsv.write("%s\n" % meta)

    board.save(fp=sprite_fp)
    # close the file handle
    meta_tsv.close()

    return sprite_fp, metadata_fp


# CLASSES

class Timeout(threading.Thread):
    """Initialize time out timer which raises an exception when the deadline is exceeded."""

    def __init__(self, timeout: int, thread_id: int = None, name: str = None, func_name=None):
        """Initialize timer.
        :param timeout: int, time in seconds
        """
        threading.Thread.__init__(self)
        self.threadID = thread_id
        self.name = name

        self._timeout = timeout
        self._time = 0
        self._run = False
        if func_name is not None:
            self._message = "function `%s` exceeded deadline." % func_name
        else:
            self._message = "deadline exceeded: `%d` seconds" % timeout

    def run(self):
        self._run = True
        while self._run:
            time.sleep(1)
            self._time += 1
            if self._time >= self._timeout:
                raise tf.errors.DeadlineExceededError(None, None, message=self._message)

    def stop(self):
        self._run = False
        self._time = 0


class AttrDict(Mapping):
    """A class to convert a nested Dictionary into an object with key-values
    accessibly using attribute notation (AttributeDict.attribute) instead of
    key notation (Dict["key"]). This class recursively sets Dicts to objects,
    allowing you to recurse down nested dicts (like: AttributeDict.attr.attr)
    """
    def __init__(self, **entries):
        for key, value in entries.items():
            # replace dashes by underscores JIC
            key = key.replace('-', '_')
            if type(value) is dict:
                self.__dict__[key] = AttrDict(**value)
            else:
                self.__dict__[key] = value

    def __iter__(self):
        for k in self.__dict__:
            yield k

    def __len__(self):
        return len(self.__dict__)

    def __str__(self):
        return self.__dict__.__str__()

    def __repr__(self):
        return self.__dict__.__repr__()

    def __getitem__(self, key):
        """
        Provides dict-style access to attributes
        """
        return getattr(self, key)


# FUNCTIONS

def plot_images(images: list, labels: list,
                prediction=None,
                cols=4,
                target_shape=None):
    # we want to plot this in grid of shape (?, 4) by default
    if prediction is not None:
        assert len(images) == len(labels) == len(prediction)
    else:
        assert len(images) == len(labels)

    assert not len(images) % cols

    rows = len(images) // cols
    # Create figure
    fig, axes = plt.subplots(rows, cols)
    fig.subplots_adjust(hspace=.3, wspace=.5)

    for i, ax in enumerate(axes.flat):
        if target_shape is not None:
            ax.imshow(images[i].reshape(target_shape), cmap='binary')
        else:
            ax.imshow(images[i], cmap='binary')

        # Show true and predicted classes
        if prediction is None:
            xlabel = "True: {0}".format(labels[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(labels[i], prediction[i])

        # Show the classes as the label on x_axis
        ax.set_xlabel(xlabel)

        # Remove the ticks from the plot
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
