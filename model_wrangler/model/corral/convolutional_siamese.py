"""Module sets up Convolutional Feedforward model"""

import tensorflow as tf

from model_wrangler.architecture import BaseArchitecture
from model_wrangler.model.layers import (
    append_dropout, append_batchnorm, append_dense, append_conv,
    append_maxpooling
)
from model_wrangler.model.losses import loss_sigmoid_ce


class ConvolutionalSiameseModel(BaseArchitecture):
    """Convolutional siamese model that has a
    couple convolutional layers and a couple of dense
    layers leading up to an output
    """

    # pylint: disable=too-many-instance-attributes

    def _conv_layer(self, in_layers, layer_param):
        layer_stack = [in_layers]

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

    def build_embedder(self, in_layer, hidden_params):
        """Build a stack of layers for mapping an input to an embedding"""

        layer_stack = [in_layer]
        for idx, layer_param in enumerate(hidden_params):
            with tf.variable_scope('conv_layer_{}/'.format(idx), reuse=tf.AUTO_REUSE):
                layer_stack.append(self._conv_layer(layer_stack[-1], layer_param))

        # Force unit-norm
        flat = tf.contrib.layers.flatten(layer_stack[-1])
        norm = tf.norm(flat, ord='euclidean', axis=1, keepdims=True, name='norm')
        layer_stack.append(tf.divide(flat, norm, name='embed_norm'))

        return layer_stack[-1]

    def setup_layers(self, params):

        #
        # Load params
        #

        in_sizes = params.get('in_sizes', [])
        hidden_params = params.get('hidden_params', [])
        num_out = params.get('num_out', 1)

        if len(in_sizes) != 2:
            raise AttributeError('Siamese network needs exactly 2 inputs') 

        #
        # Build model
        #

        in_layers = [
            tf.placeholder("float", name="input_{}".format(idx), shape=[None] + in_size)
            for idx, in_size in enumerate(in_sizes)
        ]

        with tf.variable_scope('encoder/', reuse=tf.AUTO_REUSE) as encode_scope:
            self.embed = self.build_embedder(in_layers[0], hidden_params)
            embed2 = self.build_embedder(in_layers[1], hidden_params)


        # Add final embedding layers
        embed_diff = tf.contrib.layers.flatten(self.embed - embed2)

        with tf.variable_scope('decoder/'):

            decode_coeffs = [
                tf.Variable(tf.ones([embed_diff.get_shape()[1], 1]), name="coeff_{}".format(idx))
                for idx in range(num_out)
            ]

            decode_ints = [
                tf.Variable(tf.zeros([1, ]), name="intercept_{}".format(idx))
                for idx in range(num_out)
            ]

            out_layers_preact = [
                tf.add(tf.matmul(embed_diff, coeff), intercept, name="output_{}".format(idx))
                for idx, (coeff, intercept) in enumerate(zip(decode_coeffs, decode_ints))
            ]

            out_layers = [tf.sigmoid(layer) for layer in out_layers_preact]

        target_layers = [
            tf.placeholder("float", name="target_{}".format(idx), shape=[None, 1])
            for idx in range(num_out)
        ]

        #
        # Set up loss
        #

        loss = tf.reduce_sum(
            [loss_sigmoid_ce(*pair) for pair in zip(out_layers_preact, target_layers)]
        )

        return in_layers, out_layers, target_layers, loss
