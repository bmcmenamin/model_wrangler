"""Module sets up Dense Autoencoder model"""

# pylint: disable=R0914 

import tensorflow as tf

from model_wrangler.architecture import BaseArchitecture
from model_wrangler.model.layers import (
    append_dropout, append_batchnorm, append_dense, append_categorical
)
from model_wrangler.model.losses import loss_softmax_ce, loss_crossgroup_bias

class DebiasedClassifier(BaseArchitecture):
    """Dense Feedforward with corrections for group bias"""

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

        debias_weight = params.get('debias_weight', None)


        #
        # Build model
        #

        in_layers = [
            tf.placeholder("float", name="input_{}".format(idx), shape=[None, in_size])
            for idx, in_size in enumerate(in_sizes)
        ]

        group_indexes = tf.cast(in_layers[-1], tf.int32)

        flat_input = tf.concat(
            [tf.contrib.layers.flatten(layer) for layer in in_layers[:-1]],
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

        embeds = [append_dense(self, layer_stack[-1], embed_params, 'embed_{}'.format(idx))]

        out_layer_preact = [
            append_dense(self, embeds[-1], dict(num_units=out_size), 'preact_{}'.format(idx))
            for idx, out_size in enumerate(out_sizes)
        ]

        out_layers = [
            append_categorical(self, layer, {}, name='output_{}'.format(idx))
            for idx, layer in enumerate(out_layer_preact)            
        ]

        target_layers = [
            tf.placeholder("float", name="target_{}".format(idx), shape=[None, out_size])
            for idx, out_size in enumerate(out_sizes)
        ]

        #
        # Set up loss
        #

        loss_pred = tf.reduce_sum(
            [loss_softmax_ce(*pair) for pair in zip(out_layer_preact, target_layers)]
        )

        if debias_weight and debias_weight > 0.0:
            loss_bias = tf.reduce_mean(
                [
                    loss_crossgroup_bias(tf.sigmoid(pair[0]), pair[1], group_indexes)
                    for pair in zip(out_layer_preact, target_layers)
                ]
            )

            loss = loss_pred + (debias_weight * tf.sqrt(loss_bias))
        else:
            loss = loss_pred

        tb_scalars = {
            'embed_l1': tf.reduce_mean(tf.abs(embeds[0])),
            'embed_mean': tf.reduce_mean(embeds[0])
        }

        return in_layers, out_layers, target_layers, embeds, loss, tb_scalars
