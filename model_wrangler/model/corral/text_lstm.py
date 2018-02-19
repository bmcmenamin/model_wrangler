"""Module sets up Dense Autoencoder model"""

# pylint: disable=R0914 
import numpy as np
import tensorflow as tf

from model_wrangler.model.text_tools import TextProcessor

from model_wrangler.architecture import BaseTextArchitecture
from model_wrangler.model.layers import append_dense, append_bidir_lstm_stack
from model_wrangler.model.losses import loss_softmax_ce

class TextLstmModel(BaseTextArchitecture):
    """Dense Feedforward"""

    # pylint: disable=too-many-instance-attributes

    def setup_training_step(self, params):
        """Set up loss and training step"""

        # Import params
        learning_rate = params.get('learning_rate', 0.01)

        optimizer = tf.train.RMSPropOptimizer(learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = optimizer.minimize(self.loss)

        return train_step

    def setup_layers(self, params):
        """Build all the model layers"""

        #
        # Load params
        #

        in_sizes = params.get('in_sizes', [])
        recurr_params = params.get('recurr_params', [])
        out_sizes = params.get('out_sizes', [])

        pad_len = params.get('max_string_size', 256)
        self.text_map = TextProcessor(pad_len=pad_len)

        _func_str_to_int = lambda x_list: np.vstack([
            np.array(self.text_map.string_to_ints(x, use_pad=True))
            for x in x_list
        ])

        _func_char_to_int = lambda x_list: np.vstack([
            np.array(self.text_map.string_to_ints(x, use_pad=False)[0])
            for x in x_list
        ])

        _func_int_to_str = lambda x_list: [
            self.text_map.ints_to_string(x)
            for x in x_list
        ]

        #
        # Build model
        #

        in_layers = [
            tf.placeholder("string", name="input_{}".format(idx), shape=[None,])
            for idx, _ in enumerate(in_sizes)
        ]

        in_layers_int = [
            tf.py_func(_func_str_to_int, [layer], tf.int64)
            for layer in in_layers
        ]

        for l1, l2 in zip(in_layers, in_layers_int):
            new_shape = l1.get_shape().as_list()
            l2.set_shape([new_shape[0], self.text_map.pad_len])

        # Add layers on top of each input
        layer_stacks = {}
        for idx_source, in_layer in enumerate(in_layers_int):

            with tf.variable_scope('source_{}'.format(idx_source)):
                layer_stacks[idx_source] = [self.make_onehot_encode_layer(in_layer)]

                with tf.variable_scope('lstm_stack'):
                    layer_stacks[idx_source].append(
                        append_bidir_lstm_stack(self, layer_stacks[idx_source][-1], recurr_params, 'lstm')
                    )

        # Flatten/concat output inputs from each convolutional stack

        embeds = tf.concat([
            tf.contrib.layers.flatten(layer_stack[-1])
            for layer_stack in layer_stacks.values()
        ], axis=-1)

        out_layers_preact = [
            append_dense(
                self,
                embeds,
                {'num_units': self.text_map.num_chars + 2},
                'preact_{}'.format(idx)
            )
            for idx, _ in enumerate(out_sizes)
        ]

        out_layers = [
            tf.py_func(_func_int_to_str, [
                self.make_onehot_decode_layer(out, probabilistic=True)
            ], tf.string)
            for out in out_layers_preact
        ]

        target_layers = [
            tf.placeholder("string", name="target_{}".format(idx), shape=[None,])
            for idx, _ in enumerate(out_sizes)
        ]

        target_layers_int = [
            tf.py_func(_func_char_to_int, [targ], tf.int64)
            for targ in target_layers
        ]

        for l1, l2 in zip(target_layers, target_layers_int):
            new_shape = l1.get_shape().as_list()
            l2.set_shape([new_shape[0], 1])

        target_layers_onehot = [
            self.make_onehot_encode_layer(targ)
            for targ in target_layers_int
        ]

        #
        # Set up loss
        #

        loss = tf.reduce_sum(
            [loss_softmax_ce(*pair) for pair in zip(out_layers_preact, target_layers_onehot)]
        )

        return in_layers, out_layers, target_layers, embeds, loss
