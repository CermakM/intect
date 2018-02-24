"""Module containing tools to handle data sets."""

import typing

from tensorflow.contrib.keras import preprocessing

from . import utils


class Dataset(object):
    """Base Dataset class."""

    def __init__(self,
                 images: typing.Generator,
                 labels: typing.Generator,
                 classes: dict,
                 n_samples: int,
                 target_size: tuple,
                 img_shape: tuple,
                 ):
        if not isinstance(images, typing.Generator):
            raise TypeError("`images` argument must be of type %s" % typing.Generator)

        if not isinstance(labels, typing.Generator):
            raise TypeError("`labels` argument must be of type %s" % typing.Generator)

        assert sum(1 for _ in images) == sum(1 for _ in labels), "`images` shape %s does not fit `labels` shape %s"

        self._images = images
        self._labels = labels
        self._classes = classes
        self._samples = n_samples
        self._target_size = target_size
        self._img_shape = img_shape

    def __len__(self):
        return self._samples

    def __str__(self):
        description = "Number of images: {s._samples}\nNumber of labels: {s._samples}\n" \
                      "Image shape: {s._img_shape}"
        return description.format(s=self)

    @property
    def images(self) -> typing.Generator:
        return self._images

    @property
    def labels(self) -> typing.Generator:
        return self._labels

    @property
    def classes(self) -> dict:
        return self._classes

    @property
    def empty(self) -> bool:
        return self._samples == 0

    @property
    def img_shape(self) -> tuple:
        return self._img_shape

    @classmethod
    def from_directory(cls, path: str):
        """Traverse directory and load images and labels."""

        # This loads infinite directory generator
        flow = preprocessing.image.ImageDataGenerator().flow_from_directory(path)

        images, labels = utils.flow_from_directory(path)

        return cls(images, labels, classes=flow.class_indices, n_samples=flow.samples,
                   target_size=flow.target_size, img_shape=flow.image_shape)

    def describe(self):
        print(self.__str__())
