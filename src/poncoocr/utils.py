"""Module containing common utility functions."""

import numpy as np
import typing

from collections.abc import Mapping
from matplotlib.pyplot import plot as plt
from tensorflow.contrib.keras import preprocessing


# CLASSES

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

    def __getitem__(self, key):
        """
        Provides dict-style access to attributes
        """
        return getattr(self, key)


# FUNCTIONS

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
