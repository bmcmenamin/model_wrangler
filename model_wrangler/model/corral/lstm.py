"""Module sets up Dense Autoencoder model"""

# pylint: disable=R0914 
import numpy as np
import tensorflow as tf

from model_wrangler.architecture import BaseArchitecture
from model_wrangler.model.layers import append_dense, append_lstm_stack
from model_wrangler.model.losses import loss_mse

class LstmModel(BaseArchitecture):
    """Dense Feedforward"""

    # pylint: disable=too-many-instance-attributes

    def setup_training_step(self, params):
        """Set up loss and training step"""

        learning_rate = params.get('learning_rate', 0.001)
        optimizer = tf.train.RMSPropOptimizer(learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            capped_grads = [
                (tf.clip_by_value(g, -1.0, 1.0), v)
                for g, v in optimizer.compute_gradients(self.loss)
            ]
            train_step = optimizer.apply_gradients(capped_grads)

        return train_step

    def setup_layers(self, params):
        """Build all the model layers"""

        #
        # Load params
        #

        in_sizes = params.get('in_sizes', [])
        recurr_params = params.get('recurr_params', [])
        out_sizes = params.get('out_sizes', [])

        #
        # Build model
        #

        in_layers = [
            tf.placeholder("float", name="input_{}".format(idx), shape=[None] + in_size)
            for idx, in_size in enumerate(in_sizes)
        ]

        # Add layers on top of each input
        layer_stacks = {}
        for idx_source, in_layer in enumerate(in_layers):

            with tf.variable_scope('source_{}'.format(idx_source)):
                layer_stacks[idx_source] = [in_layer]

                with tf.variable_scope('lstm_stack'):
                    layer_stacks[idx_source].append(
                        append_lstm_stack(self, layer_stacks[idx_source][-1], recurr_params, 'lstm')
                    )

        embeds = tf.concat([
            tf.contrib.layers.flatten(layer_stack[-1])
            for layer_stack in layer_stacks.values()
        ], axis=-1)


        out_layer_preact = [
            tf.expand_dims(
                append_dense(self, embeds, {'num_units': out_size}, 'preact_{}'.format(idx)),
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

        tb_scalars = {}
        return in_layers, out_layers, target_layers, embeds, loss, tb_scalars
