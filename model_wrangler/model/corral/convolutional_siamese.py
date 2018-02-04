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

        if len(in_sizes) != 2:
            raise AttributeError('Siamese network needs exactly 2 inputs') 

        #
        # Build model
        #

        in_layers = [
            tf.placeholder("float", name="input_{}".format(idx), shape=[None] + in_size)
            for idx, in_size in enumerate(in_sizes)
        ]

        with tf.variable_scope('encoder/', reuse=tf.AUTO_REUSE):
            embeds = [
                self.build_embedder(in_layers[0], hidden_params),
                self.build_embedder(in_layers[1], hidden_params)
            ]

        with tf.variable_scope('decoder/'):
            #decode_scale = tf.Variable(tf.ones([1, 1]), name="scale_{}".format(idx))
            decode_int = tf.Variable(tf.zeros([1, ]), name="intercept")
            out_layers_preact = decode_int + tf.reduce_sum(
                tf.multiply(*embeds), 1, keepdims=True
                )

            out_layers = [tf.sigmoid(out_layers_preact)]

        target_layers = [
            tf.placeholder("float", name="target", shape=[None, 1])
        ]

        #
        # Set up loss
        #

        loss = tf.reduce_sum(loss_sigmoid_ce(out_layers_preact, target_layers[0]))

        return in_layers, out_layers, target_layers, embeds, loss
