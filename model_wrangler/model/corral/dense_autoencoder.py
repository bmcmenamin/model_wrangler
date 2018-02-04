"""Module sets up Dense Autoencoder model"""

# pylint: disable=R0914

import tensorflow as tf

from model_wrangler.architecture import BaseArchitecture
from model_wrangler.model.layers import append_dropout, append_batchnorm, append_dense, fit_to_shape
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
        out_sizes = in_sizes #params.get('out_sizes', [])

        #
        # Build model
        #

        in_layers = [
            tf.placeholder("float", name="input_{}".format(idx), shape=[None, in_size])
            for idx, in_size in enumerate(in_sizes)
        ]

        layer_stack = [in_layers[0]]

        # Encoding layers
        for idx, layer_param in enumerate(encoding_params):
            with tf.variable_scope('encoding_layer_{}'.format(idx)):
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
        with tf.variable_scope('bottleneck_layer'):
            layer_stack.append(
                append_dense(self, layer_stack[-1], embed_params, 'dense')
                )

            layer_stack.append(
                append_batchnorm(self, layer_stack[-1], embed_params, 'batchnorm')
                )

            layer_stack.append(
                append_dropout(self, layer_stack[-1], embed_params, 'dropout')
                )

            self.embed = [layer_stack[-1]]

        # Decoding layers
        for idx, layer_param in enumerate(decoding_params):
            with tf.variable_scope('decdoding_layer{}'.format(idx)):
                layer_stack.append(
                    append_dense(self, layer_stack[-1], layer_param, 'dense')
                    )

        out_layers = [
            fit_to_shape(self, layer_stack[-1], {'target_shape': [None, out_size]}, 'recon')
            for idx, out_size in enumerate(out_sizes)
        ]

        target_layers = [
            tf.placeholder("float", name="target_{}".format(idx), shape=[None, out_size])
            for idx, out_size in enumerate(out_sizes)
        ]

        #
        # Set up loss
        #

        loss = tf.reduce_sum(
            [loss_mse(*pair) for pair in zip(out_layers, target_layers)]
        )

        return in_layers, out_layers, target_layers, loss
