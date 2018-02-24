"""Module containing common utility functions."""

import numpy as np
import typing

from tensorflow.contrib.keras import preprocessing


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
