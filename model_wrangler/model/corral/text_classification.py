"""Module sets up Convolutional Text Classifier"""

import numpy as np
import tensorflow as tf

from model_wrangler.model.text_tools import TextProcessor

from model_wrangler.architecture import BaseArchitecture
from model_wrangler.model.layers import (
    append_dropout, append_batchnorm, append_dense, append_conv,
    append_maxpooling
)
from model_wrangler.model.losses import loss_softmax_ce


class TextClassificationModel(BaseArchitecture):
    """Convolutional feedforward model that has a
    couple convolutional layers and a couple of dense
    layers leading up to an output
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, params):
        self.text_map = None
        super().__init__(params)

    def make_onehot_encode_layer(self, in_layer):
        """Return a layer that one-hot encodes an int layer

        Args:
            in_layer: A layer consisting of integer values

        Returns:
            a new layer that has one-hot encoded the integers
        """

        onehot_layer = tf.one_hot(
            tf.to_int32(in_layer),
            len(self.text_map.good_chars) + 2,
            axis=-1
        )

        return onehot_layer

    @staticmethod
    def make_onehot_decode_layer(in_layer):
        """Return a layer takes one-hot encoded layer to int
        Args:
            in_layer: A of one-hot endcoded values

        Returns:
            a new layer with the index of the largest positive value
        """

        out_layer = tf.argmax(in_layer, axis=-1)
        return out_layer

    def _conv_layer(self, in_layer, layer_param):
        layer_stack = [tf.to_float(in_layer)]

        layer_stack.append(
            append_conv(self, layer_stack[-1], layer_param, 'conv')
            )

        layer_stack.append(
            append_maxpooling(self, layer_stack[-1], layer_param, 'maxpool')
            )

        layer_stack.append(
            append_batchnorm(self, layer_stack[-1], layer_param, 'batchnorm')
            )

        layer_stack.append(
            append_dropout(self, layer_stack[-1], layer_param, 'dropout')
            )

        return layer_stack[-1]

    def setup_layers(self, params):

        #
        # Load params
        #

        num_in = params.get('num_inputs', 1)
        pad_len = params.get('pad_length', 256)
        hidden_params = params.get('hidden_params', [])
        embed_params = params.get('embed_params', [])
        out_sizes = params.get('out_sizes', [])

        self.text_map = TextProcessor(pad_len=pad_len)

        #
        # Build model
        #

        in_layers = [
            tf.placeholder("string", name="input_{}".format(idx), shape=[None,])
            for idx in range(num_in)
        ]

        _func = lambda x_list: np.vstack([
            np.array(self.text_map.string_to_ints(x))
            for x in x_list
        ])

        in_layers_int = [
            tf.py_func(_func, [layer], tf.int64)
            for layer in in_layers
        ]

        for l1, l2 in zip(in_layers, in_layers_int):
            new_shape = l1.get_shape().as_list()
            l2.set_shape([new_shape[0], pad_len])

        # Add layers on top of each input
        layer_stacks = {}
        for idx_source, in_layer in enumerate(in_layers_int):
            with tf.variable_scope('source_{}'.format(idx_source)):
                layer_stacks[idx_source] = [self.make_onehot_encode_layer(in_layer)]
                for idx_layer, layer_param in enumerate(hidden_params):
                    with tf.variable_scope('params_{}'.format(idx_layer)):
                        layer_stacks[idx_source].append(
                            self._conv_layer(layer_stacks[idx_source][-1], layer_param)
                        )

        # Flatten/concat output inputs from each convolutional stack
        embedding_layer = tf.concat([
            tf.contrib.layers.flatten(layer_stack[-1])
            for layer_stack in layer_stacks.values()
        ], axis=-1)

        # Add final dense layer to sum it up
        out_layer_preact = [
            append_dense(self, embedding_layer, embed_params, 'preact_output_{}'.format(idx))
            for idx, out_size in enumerate(out_sizes)
        ]

        out_layers = [
            tf.nn.softmax(layer, name='output_{}'.format(idx))
            for idx, layer in enumerate(out_layer_preact)
        ]

        target_layers = [
            tf.placeholder("float", name="target_{}".format(idx), shape=[None, out_size])
            for idx, out_size in enumerate(out_sizes)
        ]

        #
        # Set up loss
        #

        loss = tf.reduce_sum(
            [loss_softmax_ce(*pair) for pair in zip(out_layer_preact, target_layers)]
        )

        return in_layers, out_layers, target_layers, loss
