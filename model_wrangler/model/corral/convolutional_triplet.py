"""Module sets up Convolutional Triplet model"""

import tensorflow as tf

from model_wrangler.architecture import BaseArchitecture
from model_wrangler.model.layers import (
    append_dropout, append_batchnorm, append_dense, append_conv,
    append_maxpooling
)

from model_wrangler.model.losses import siamese_embedding_loss

class ConvolutionalTripletModel(BaseArchitecture):
    """Convolutional triplet model that has a
    couple convolutional layers and a couple of dense
    layers leading up to an embedding layer
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
            with tf.variable_scope('conv_layer_{}/'.format(idx)):
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
        num_targets = params.get('num_targets', 1)

        if len(in_sizes) != 1:
            raise AttributeError('Embedding net takes one input only!') 

        #
        # Build model
        #

        in_layers = [
            tf.placeholder("float", name="input_{}".format(idx), shape=[None] + in_size)
            for idx, in_size in enumerate(in_sizes)
        ]

        with tf.variable_scope('encoder/', reuse=tf.AUTO_REUSE):
            embeds_per_input = [
                self.build_embedder(layer, hidden_params)
                for layer in in_layers
            ]
            embeds = tf.concat(embeds_per_input, axis=-1)

        out_layers = embeds

        target_layers = [
            tf.placeholder("float", name="target_{}".format(idx), shape=[None, 1])
            for idx in range(num_targets)
        ]

        loss = tf.reduce_sum([
            tf.contrib.losses.metric_learning.triplet_semihard_loss(tf.reshape(targ, [-1]), embeds)
            for targ in target_layers
        ])

        tb_scalars = {}
        return in_layers, out_layers, target_layers, embeds, loss, tb_scalars
