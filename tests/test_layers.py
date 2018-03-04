"""Tests for model module."""

import typing
import pytest
import unittest

import tensorflow as tf

# The imports will need to be fixed to test installed version instead of the dev one
from . import common
from src import poncoocr as pcr


TEST_DATASET = pcr.dataset.Dataset.from_directory(common.TEST_DATASET_PATH)


class TestLayers(unittest.TestCase):

    def test_layer(self):
        layer = pcr.layers.Layer(input_data=TEST_DATASET.batch(32),
                                 output_channels=10,  # just keep the layer simple
                                 activation=tf.nn.relu,
                                 )
        self.assertIsInstance(layer, pcr.layers.Layer)

    def test_shape(self):
        layer = pcr.layers.Layer(input_data=TEST_DATASET.batch(32),
                                 output_channels=10,  # just keep the layer simple
                                 activation=tf.nn.relu,
                                 )
        self.assertEqual(layer.shape, (None, 32, 32, 10))

    def test_layer_activate(self):
        """Test base class layer activation raises not implemented error."""
        layer = pcr.layers.Layer(input_data=TEST_DATASET.batch(32),
                                 output_channels=10,  # just keep the layer simple
                                 activation=tf.nn.relu,
                                 )
        with pytest.raises(expected_exception=NotImplementedError):
            _ = layer.activate()

    def test_conv_layer(self):
        """Test convolutional layer creation."""
        conv_layer = pcr.layers.ConvLayer(input_data=TEST_DATASET.batch(32),
                                          output_channels=10,  # just keep the layer simple
                                          activation=tf.nn.relu,
                                          )
        self.assertIsInstance(conv_layer, pcr.layers.ConvLayer)

    def test_fcl_layer(self):
        """Test fully connected layer creation."""
        fcl_layer = pcr.layers.FullyConnectedLayer(
            input_data=TEST_DATASET.batch(32),
            output_channels=10,  # just keep the layer simple
            activation=tf.nn.relu,
            name="test-fcl-layer",
        )
        self.assertIsInstance(fcl_layer, pcr.layers.FullyConnectedLayer)

    def test_conv_layer_activate(self):
        """Test convolutional layer activation."""
        conv_layer = pcr.layers.ConvLayer(input_data=TEST_DATASET.batch(32),
                                          output_channels=10,  # just keep the layer simple
                                          activation=tf.nn.relu,
                                          name="test-conv-layer",
                                          )
        activated_layer = conv_layer.activate()
        self.assertIsInstance(activated_layer, tf.Tensor)

    def test_fcl_layer_activate(self):
        """Test fully connected layer activation."""
        fcl_layer = pcr.layers.FullyConnectedLayer(
            input_data=TEST_DATASET.batch(32),
            output_channels=10,  # just keep the layer simple
            activation=tf.nn.relu,
            name="test-fcl-layer",
        )
        activated_layer = fcl_layer.activate()
        self.assertIsInstance(activated_layer, tf.Tensor)
