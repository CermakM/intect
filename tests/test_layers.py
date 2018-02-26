"""Tests for model module."""

import typing
import pytest
import unittest

import tensorflow as tf

# The imports will need to be fixed to test installed version instead of the dev one
from . import common
from src import poncoocr as pcr

test_dataset = pcr.dataset.Dataset.from_directory(common.TEST_DATASET_PATH)


class TestLayers(unittest.TestCase):

    def test_layer(self):
        layer = pcr.layers.Layer(input_data=test_dataset.images,
                                 input_channels=test_dataset.img_shape[2],  # (width, hight, channels)
                                 output_channels=10,  # just keep the layer simple
                                 activation=tf.nn.relu,
                                 name="test-layer",
                                 )
        self.assertIsInstance(layer, pcr.layers.Layer)

    def test_layer_activate(self):
        layer = pcr.layers.Layer(input_data=test_dataset.images,
                                 input_channels=test_dataset.img_shape[2],  # (width, hight, channels)
                                 output_channels=10,  # just keep the layer simple
                                 activation=tf.nn.relu,
                                 name="test-layer",
                                 )
        with pytest.raises(expected_exception=NotImplementedError):
            _ = layer.activate()

    def test_conv_layer(self):
        conv_layer = pcr.layers.ConvLayer(input_data=test_dataset.images,
                                          input_channels=test_dataset.img_shape[2],  # (width, hight, channels)
                                          output_channels=10,  # just keep the layer simple
                                          activation=tf.nn.relu,
                                          name="test-conv-layer",
                                          )
        self.assertIsInstance(conv_layer, pcr.layers.ConvLayer)

    def test_fcl_layer(self):
        fcl_layer = pcr.layers.FullyConnectedLayer(
            input_data=test_dataset.images,
            input_channels=test_dataset.img_shape[2],  # (width, hight, channels)
            output_channels=10,  # just keep the layer simple
            activation=tf.nn.relu,
            name="test-fcl-layer",
        )
        self.assertIsInstance(fcl_layer, pcr.layers.FullyConnectedLayer)

    def test_conv_layer_activate(self):
        conv_layer = pcr.layers.ConvLayer(input_data=test_dataset.images,
                                          input_channels=test_dataset.img_shape[2],  # (width, hight, channels)
                                          output_channels=10,  # just keep the layer simple
                                          activation=tf.nn.relu,
                                          name="test-conv-layer",
                                          )
        activated_layer = conv_layer.activate()
        self.assertIsInstance(activated_layer, tf.Tensor)

    def test_fcl_layer_activate(self):
        fcl_layer = pcr.layers.FullyConnectedLayer(
            input_data=test_dataset.images,
            input_channels=test_dataset.img_shape[2],  # (width, hight, channels)
            output_channels=10,  # just keep the layer simple
            activation=tf.nn.relu,
            name="test-fcl-layer",
        )
        activated_layer = fcl_layer.activate()
        self.assertIsInstance(activated_layer, tf.Tensor)
