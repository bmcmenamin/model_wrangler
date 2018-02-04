"""Module sets up Dense Autoencoder model"""

# pylint: disable=R0914 

import tensorflow as tf

from model_wrangler.architecture import BaseArchitecture
from model_wrangler.model.layers import append_dropout, append_batchnorm, append_dense
from model_wrangler.model.losses import loss_softmax_ce

class DenseFeedforwardModel(BaseArchitecture):
    """Dense Feedforward"""

    # pylint: disable=too-many-instance-attributes

    def setup_layers(self, params):
        """Build all the model layers"""

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
            tf.placeholder("float", name="input_{}".format(idx), shape=[None, in_size])
            for idx, in_size in enumerate(in_sizes)
        ]

        flat_input = tf.concat(
            [tf.contrib.layers.flatten(layer) for layer in in_layers],
            name='flat_inputs',
            axis=-1
        )

        layer_stack = [flat_input]
        for idx, layer_param in enumerate(hidden_params):
            with tf.variable_scope('params_{}'.format(idx)):

                layer_stack.append(
                    append_dense(self, layer_stack[-1], layer_param, 'dense')
                    )

                layer_stack.append(
                    append_batchnorm(self, layer_stack[-1], layer_param, 'batchnorm')
                    )

                layer_stack.append(
                    append_dropout(self, layer_stack[-1], layer_param, 'dropout')
                    )

        embeds = [layer_stack[-1]]

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

        return in_layers, out_layers, target_layers, embeds, loss
