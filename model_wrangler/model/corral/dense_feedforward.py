"""Module sets up Dense Autoencoder model"""

# pylint: disable=R0914 

import tensorflow as tf

from model_wrangler.architecture import BaseArchitecture
from model_wrangler.model.layers import append_dropout, append_batchnorm, append_dense
from model_wrangler.model.losses import loss_sigmoid_ce

"""
hidden_params = [
    {
        'num_units': 10,
        'bias': True,
        'activation': 'relu',
        'activity_reg': {'l1': 0.1},
        'dropout_rate': 0.1
    },
    {
        'num_units': 10,
        'bias': True,
        'activation': 'relu',
        'activity_reg': {'l1': 0.1}
        'dropout_rate': 0.1
    }
]

embed_params =     {
    'num_units': 10,
    'bias': True,
    'activation': None,
    'activity_reg': None
    'dropout_rate': None
}
"""


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

        layer_stack = []
        for idx, layer_param in enumerate(hidden_params):
            with tf.name_scope('params_{}'.format(idx)):

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
            append_dense(self, layer_stack[-1], embed_params, 'preact')
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
            [loss_sigmoid_ce(*pair) for pair in zip(target_layers, out_layer_preact)]
        )

        return in_layers, out_layers, target_layers, loss
