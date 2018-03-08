"""Module containing tools to handle data sets."""

import typing

import numpy as np
import tensorflow as tf


class Dataset(object):
    """Custom data set loader using DirectoryIterator class."""

    def __init__(self):
        pass

    @staticmethod
    def from_directory(directory: str,
                       batch_size=1,
                       target_size=(32, 32)) -> tf.data.Dataset:
        """Loads a dataset from the given directory using DirectoryIterator.

        returns: `Dataset`
        """
        dir_iter = DirectoryIterator(directory, batch_size, target_size)
        features, labels = dir_iter.features, dir_iter.labels

        return tf.data.Dataset.from_tensor_slices((features, labels))


class DirectoryIterator:
    """Base Dataset class."""

    def __init__(self, directory: str, batch_size=1, target_size=(32, 32)):
        """Traverse directory and load images and labels."""

        # This loads infinite directory generator
        self._flow = tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory(
            directory=directory,
            batch_size=batch_size,
            target_size=target_size
        )

        self._classes = self._flow.class_indices,
        self._samples, = self._flow.samples,
        self._target_size = self._flow.target_size,
        self._img_shape = self._flow.image_shape

        self._features, self._labels = list(zip(*[self._flow.next() for _ in range(self._samples)]))

    def __len__(self):
        return self._samples

    def __str__(self):
        description = "<%s>  images: {s._samples}  shapes: {s._img_shape}" % type(self)

        return description.format(s=self)

    def __iter__(self):
        return self

    def __next__(self):
        return self._flow.next()

    @property
    def features(self) -> typing.Sequence:
        """Property holding features of the loaded data set as a np.array."""
        return np.concatenate(self._features)

    @property
    def labels(self) -> typing.Sequence:
        """Property holding labels of the loaded data set as a np.array."""
        return np.concatenate(self._labels)

    @property
    def classes(self):
        return self._classes

    @property
    def empty(self) -> bool:
        return self._samples == 0

    @property
    def img_shape(self) -> tuple:
        return self._img_shape

    def describe(self):
        print(self.__str__())
