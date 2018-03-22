"""Module containing tools to handle data sets."""

import typing

import numpy as np
import tensorflow as tf


class Dataset(object):
    """Custom data set loader using DirectoryIterator class."""

    def __init__(self,
                 features,
                 labels,
                 classes: dict = None,
                 shape: typing.Sequence = None):
        """Initialize Dataset."""
        self._features = features
        self._labels = labels
        self._classes = classes
        self._shape = shape

    def __len__(self):
        return len(self._labels)

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    @property
    def classes(self):
        return self._classes

    @classmethod
    def from_directory(cls,
                       directory: str,
                       batch_size=1,
                       normalize=True,
                       mode='grayscale',
                       target_size=(32, 32)):
        """Loads a dataset from the given directory using DirectoryIterator.

        returns: `Dataset`
        """
        dir_iter = DirectoryIterator(directory, batch_size, target_size,
                                     normalize=normalize, mode=mode)
        features, labels = dir_iter.features, dir_iter.labels

        return cls(features, labels, classes=dir_iter.classes, shape=dir_iter.img_shape)

    def make_one_shot_iterator(self, batch_size=None, repeat=None,
                               shuffle=True, buffer_size=None):

        dataset = tf.data.Dataset.from_tensor_slices((self._features, self._labels))
        dataset = dataset.repeat(repeat)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=buffer_size or len(self))

        if batch_size:
            dataset = dataset.batch(batch_size=batch_size)

        return dataset.make_one_shot_iterator()


class DirectoryIterator:
    """Base Dataset class."""

    def __init__(self,
                 directory: str,
                 batch_size=1,
                 target_size=(32, 32),
                 revert=False,
                 normalize=False,
                 mode='grayscale'):
        """Traverse directory and load images and labels."""

        # This loads infinite directory generator
        self._flow = tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory(
            directory=directory,
            color_mode=mode,
            batch_size=batch_size,
            target_size=target_size
        )

        self._classes, = self._flow.class_indices,
        self._samples, = self._flow.samples,
        self._target_size = self._flow.target_size,
        self._img_shape = self._flow.image_shape

        self._features, self._labels = list(zip(*[self._flow.next() for _ in range(self._samples)]))
        # convert to numpy arrays
        self._features, self._labels = np.array(self._features), np.array(self._labels)
        if normalize:
            self._features /= 255
        if revert:
            self._features = 255 - self._features

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
