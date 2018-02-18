"""Module sets up Dense Autoencoder model"""

# pylint: disable=R0914 
import numpy as np
import tensorflow as tf

from tensorflow import keras
from keras.layers import Dense, TimeDistributed

from model_wrangler.architecture import BaseArchitecture
from model_wrangler.model.layers import append_dense
from model_wrangler.model.losses import loss_mse

class LstmModel(BaseArchitecture):
    """Dense Feedforward"""

    # pylint: disable=too-many-instance-attributes

    def setup_layers(self, params):
        """Build all the model layers"""

        #
        # Load params
        #

        in_sizes = params.get('in_sizes', [])
        dense_params = params.get('dense_params', [])
        recurr_params = params.get('recurr_params', [])
        out_sizes = params.get('out_sizes', [])

        #
        # Build model
        #

        if len(in_sizes) != 1:
            raise AttributeError('Only one input allowed for LSTM network') 

        in_layers = [
            tf.placeholder("float", name="input_{}".format(idx), shape=[None] + in_size)
            for idx, in_size in enumerate(in_sizes)
        ]

        layer_stack = [in_layers[0]]

        for idx, layer_param in enumerate(dense_params):
            with tf.variable_scope('conv_{}'.format(idx)):
                layer_stack.append(
                    TimeDistributed(
                        Dense(
                            layer_param.get('num_units', 3),
                            activation=layer_param.get('activation', None),
                            use_bias=layer_param.get('bias', True)
                        ),
                        input_shape=layer_stack[-1].get_shape().as_list()[2:],
                        name='dense_{}'.format(idx)
                    )(layer_stack[-1])
                )

        for idx, layer_param in enumerate(recurr_params):

            last_layer = idx == (len(recurr_params) - 1)
            with tf.variable_scope('lstms_{}'.format(idx)):
                layer_stack.append(
                    tf.keras.layers.LSTM(
                        stateful=False,
                        return_sequences=not last_layer,
                        **layer_param
                    )(layer_stack[-1])
                )

            if last_layer:
                embeds = layer_stack[-1]

        out_layer_preact = [
            tf.expand_dims(
                append_dense(self, layer_stack[-1], {'num_units': out_size}, 'preact_{}'.format(idx)),
            1)
            for idx, out_size in enumerate(out_sizes)
        ]

        out_layers = out_layer_preact

        target_layers = [
            tf.expand_dims(
                tf.placeholder("float", name="target_{}".format(idx), shape=[None, out_size]),
            1)
            for idx, out_size in enumerate(out_sizes)
        ]

        #
        # Set up loss
        #

        loss = tf.reduce_sum(
            [loss_mse(*pair) for pair in zip(out_layer_preact, target_layers)]
        )

        return in_layers, out_layers, target_layers, embeds, loss
