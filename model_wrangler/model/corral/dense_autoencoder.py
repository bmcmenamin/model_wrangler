"""Module sets up Dense Autoencoder model"""

# pylint: disable=R0914

import tensorflow as tf

from model_wrangler.architecture import BaseArchitecture
from model_wrangler.model.layers import append_dropout, append_batchnorm, append_dense
from model_wrangler.model.losses import loss_mse


class DenseAutoencoderModel(BaseArchitecture):
    """Dense autoencoder model"""


    def setup_layers(self, params):
        """Build all the model layers"""

        #
        # Load params
        #

        in_sizes = params.get('in_sizes', [])
        encoding_params = params.get('encoding_params', [])
        embed_params = params.get('embed_params', {})
        decoding_params = params.get('decoding_params', [])
        out_sizes = params.get('out_sizes', [])

        #
        # Build model
        #

        in_layers = [
            tf.placeholder("float", name="input_{}".format(idx), shape=[None, in_size])
            for idx, in_size in enumerate(in_sizes)
        ]

        layer_stack = []

        # Encoding layers
        for idx, layer_param in enumerate(encoding_params):
            with tf.name_scope('encoding_layer_{}'.format(idx)):
                layer_stack.append(
                    append_dense(self, layer_stack[-1], layer_param, 'dense')
                    )

                layer_stack.append(
                    append_batchnorm(self, layer_stack[-1], layer_param, 'batchnorm')
                    )

                layer_stack.append(
                    append_dropout(self, layer_stack[-1], layer_param, 'dropout')
                    )

        # Bottleneck
        with tf.name_scope('bottleneck_layer'):
            layer_stack.append(
                append_dense(self, layer_stack[-1], embed_params, 'dense')
                )

            layer_stack.append(
                append_batchnorm(self, layer_stack[-1], embed_params, 'batchnorm')
                )

            layer_stack.append(
                append_dropout(self, layer_stack[-1], embed_params, 'dropout')
                )

        # Decoding layers
        for idx, layer_param in enumerate(decoding_params):
            with tf.name_scope('decdoding_layer{}'.format(idx)):
                layer_stack.append(
                    append_dense(self, layer_stack[-1], layer_param, 'dense')
                    )

                layer_stack.append(
                    append_batchnorm(self, layer_stack[-1], layer_param, 'batchnorm')
                    )

                layer_stack.append(
                    append_dropout(self, layer_stack[-1], layer_param, 'dropout')
                    )

        out_layer_preact = [
            tf.placeholder("float", name="output_{}".format(idx), shape=[None, out_size])
            for idx, out_size in enumerate(out_sizes)
        ]

        out_layers = [
            tf.sigmoid(layer, name='output_0') for layer in out_layer_preact
        ]

        target_layers = [
            tf.placeholder("float", name="target_{}".format(idx), shape=[None, out_size])
            for idx, out_size in enumerate(out_sizes)
        ]

        #
        # Set up loss
        #

        loss = tf.reduce_sum(
            [loss_mse(*pair) for pair in zip(target_layers, out_layers)]
        )

        return in_layers, out_layers, target_layers, loss


class DenseAutoencoder(ModelWrangler):
    """Dense Autoencoder
    """
    def __init__(self, in_size=10, **kwargs):
        super(DenseAutoencoder, self).__init__(
            model_class=DenseAutoencoderModel,
            in_size=in_size,
            **kwargs)
