"""Module sets up Dense Autoencoder model"""

import tensorflow as tf

from model_wrangler.architecture import BaseArchitecture
from model_wrangler.model.layers import (
    append_dropout, append_batchnorm, append_dense,
    append_conv, append_maxpooling,
    append_deconv, append_unstride, append_unpool,
    fit_to_shape
    )
from model_wrangler.model.losses import loss_mse


class ConvolutionalAutoencoderModel(BaseArchitecture):
    """Convolutionals autoencoder model"""

    # pylint: disable=too-many-instance-attributes

    def _encode_layer(self, in_layer, layer_param):

        layer_stack = [in_layer]

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


    def _decode_layer(self, in_layer, layer_param):

        layer_stack = [in_layer]

        layer_stack.append(
            append_deconv(self, layer_stack[-1], layer_param, 'deconv')
            )

        layer_stack.append(
            append_unpool(self, layer_stack[-1], layer_param, 'unpool')
            )

        layer_stack.append(
            append_unstride(self, layer_stack[-1], layer_param, 'unstride')
            )

        return layer_stack[-1]



    def setup_layers(self, params):

        #
        # Load params
        #

        in_sizes = params.get('in_sizes', [])
        encoding_params = params.get('encoding_params', [])
        embed_params = params.get('embed_params', [])
        decoding_params = params.get('decoding_params', [])
        out_sizes = in_sizes #params.get('out_sizes', [])

        if len(in_sizes) != 1:
            raise AttributeError('Siamese network needs exactly 1 input') 

        #
        # Build model
        #

        in_layers = [
            tf.placeholder("float", name="input_{}".format(idx), shape=[None] + in_size)
            for idx, in_size in enumerate(in_sizes)
        ]

        layer_stack = [in_layers[0]]

        for idx, layer_param in enumerate(encoding_params):
            with tf.variable_scope('encoding_{}'.format(idx)):
                layer_stack.append(
                    self._encode_layer(layer_stack[-1], layer_param)
                )

        with tf.variable_scope('embedding_layer'):

            # Flatten convolutional layers
            layer_stack.append(
                tf.contrib.layers.flatten(layer_stack[-1])
            )

            layer_stack.append(
                append_dense(self, layer_stack[-1], embed_params, 'embedding')
                )

            for _ in range(len(in_sizes[0]) - 1):
                layer_stack.append(
                    tf.expand_dims(layer_stack[-1], -1)
                    )

            embeds = [layer_stack[-1]]

        for idx, layer_param in enumerate(decoding_params):
            with tf.variable_scope('decoding_{}'.format(idx)):
                layer_stack.append(
                    self._decode_layer(layer_stack[-1], layer_param)
                )

        # Add final embedding layers

        out_layers = [
            fit_to_shape(self, layer_stack[-1], {'target_shape': [None] + out_size}, 'recon')
            for idx, out_size in enumerate(out_sizes)
        ]

        target_layers = [
            tf.placeholder("float", name="target_{}".format(idx), shape=[None] + out_size)
            for idx, out_size in enumerate(out_sizes)
        ]

        #
        # Set up loss
        #

        loss = tf.reduce_sum(
            [loss_mse(*pair) for pair in zip(out_layers, target_layers)]
        )

        tb_scalars = {}
        return in_layers, out_layers, target_layers, embeds, loss, tb_scalars
