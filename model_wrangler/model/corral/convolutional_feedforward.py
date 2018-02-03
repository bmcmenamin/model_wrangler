"""Module sets up Convolutional Feedforward model"""

import tensorflow as tf

from model_wrangler.architecture import BaseArchitecture
from model_wrangler.model.layers import append_dropout, append_batchnorm, append_dense, append_conv
from model_wrangler.model.losses import loss_softmax_ce


class ConvolutionalFeedforwardModel(BaseArchitecture):
    """Convolutional feedforward model that has a
    couple convolutional layers and a couple of dense
    layers leading up to an output
    """

    # pylint: disable=too-many-instance-attributes

    def setup_layers(self, params):

        #
        # Load params
        #

        in_sizes = params.get('in_sizes', [])
        hidden_params = params.get('hidden_params', [])
        embed_params = params.get('embed_params', [])
        out_sizes = params.get('out_sizes', [])

        #
        # Build model
        #

        in_layers = [
            tf.placeholder("float", name="input_{}".format(idx), shape=[None] + in_size)
            for idx, in_size in enumerate(in_sizes)
        ]

        layer_stack = [in_layers[0]]

        for idx, layer_param in enumerate(hidden_params):
            with tf.variable_scope('params_{}'.format(idx)):

                layer_stack.append(
                    append_conv(self, layer_stack[-1], layer_param, 'dense')
                    )

                layer_stack.append(
                    append_batchnorm(self, layer_stack[-1], layer_param, 'batchnorm')
                    )

                layer_stack.append(
                    append_dropout(self, layer_stack[-1], layer_param, 'dropout')
                    )

        # Flatten convolutional layers
        layer_stack.append(
            tf.contrib.layers.flatten(layer_stack[-1])
        )

        # Add final embedding layers

        out_layer_preact = [
            append_dense(self, layer_stack[-1], embed_params, 'preact_{}'.format(idx))
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
