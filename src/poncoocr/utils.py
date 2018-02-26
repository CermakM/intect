"""Module containing common utility functions."""

import numpy as np
import typing

from matplotlib.pyplot import plot as plt
from tensorflow.contrib.keras import preprocessing


# CLASSES

class AttrDict(object):
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

    def __getitem__(self, key):
        """
        Provides dict-style access to attributes
        """
        return getattr(self, key)


# FUNCTIONS

def labels_from_directory(path: str) -> typing.List["np.ndarray"]:
    """Traverses directory and returns list of subdirectories.
    Assumes Keras-like directory structure where subdirectory of the given `path`
    corresponds to a label.
    """

    im_gen = preprocessing.image.ImageDataGenerator()
    # This loads infinite directory generator
    flow = im_gen.flow_from_directory(path, batch_size=1, target_size=(32, 32))

    labels = [None] * flow.samples

    for i in range(flow.samples):
        labels[i] = next(flow)[1]

    return labels


def images_from_directory(path: str) -> typing.List["np.ndarray"]:
    """Traverses directory and returns list of valid PIL images as numpy arrays."""
    # TODO: check shape of the images
    im_gen = preprocessing.image.ImageDataGenerator()
    # This loads infinite directory generator
    flow = im_gen.flow_from_directory(path, batch_size=1, target_size=(32, 32))

    images = [None] * flow.samples

    for i in range(flow.samples):
        images[i] = next(flow)[0]

    return images


def load_data_from_directory(path: str) -> typing.Tuple["np.ndarray", "np.ndarray"]:
    """Traverses directory and returns list of tuple (image, label)."""

    im_gen = preprocessing.image.ImageDataGenerator()
    # This loads infinite directory generator
    flow = im_gen.flow_from_directory(path, batch_size=1, target_size=(32, 32))

    data = [None] * flow.samples
    for i in range(flow.samples):
        data[i] = next(flow)

    return data


def flow_from_directory(path: str) -> typing.Tuple[typing.Generator, typing.Generator]:
    """Traverses directory and returns tuple of generators (images, labels)."""

    # This loads infinite directory generator for both images and labels separately
    im_flow = preprocessing.image.ImageDataGenerator().flow_from_directory(path, batch_size=1, target_size=(32, 32))
    label_flow = preprocessing.image.ImageDataGenerator().flow_from_directory(path, batch_size=1, target_size=(32, 32))

    # Create generator expressions
    im_gen = (next(im_flow)[0] for _ in range(im_flow.samples))
    label_gen = (next(label_flow)[1] for _ in range(label_flow.samples))

    return im_gen, label_gen


def plot_images(images, labels,
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
