"""Module sets up Dense Autoencoder model"""

# pylint: disable=R0914 
import numpy as np
import tensorflow as tf

from model_wrangler.model.text_tools import TextProcessor

from model_wrangler.architecture import BaseTextArchitecture
from model_wrangler.model.layers import append_dense, append_lstm_stack
from model_wrangler.model.losses import loss_softmax_ce

class TextLstmModel(BaseTextArchitecture):
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
        in_size = params.get('win_length', 3)
        embed_size = params.get('embed_size', 64)
        recurr_params = params.get('recurr_params', [])

        self.text_map = TextProcessor(pad_len=in_size)
        self.char_embeddings = self.make_embedding_layer(embed_size=embed_size)
        vocab_size, embed_size = self.char_embeddings.get_shape().as_list()

        _func_str_to_int = lambda x_list: np.vstack([
            np.array(self.text_map.string_to_ints(x, use_pad=True))
            for x in x_list
        ])

        _func_int_to_str = lambda x_list: [
            self.text_map.ints_to_string(x)
            for x in x_list
        ]

        #
        # Build model
        #

        in_layer = tf.placeholder("string", name="input_0", shape=[None,])
        in_layer_shape = in_layer.get_shape().as_list()

        target_layer = tf.placeholder("string", name="target", shape=[None,])

        with tf.variable_scope('input'):

            in_layers_int = tf.py_func(_func_str_to_int, [in_layer], tf.int64, name='int')
            in_layers_int.set_shape([None, in_size])

            in_layers_onehot = tf.reshape(
                tf.one_hot(
                    tf.reshape(in_layers_int, [-1]),
                    vocab_size
                ),
                [-1, in_size, vocab_size],
                name='oh'
            )

            in_layers_embed = tf.reshape(
                self.get_embeddings(
                    tf.reshape(in_layers_int, [-1]),
                ),
                [-1, in_size, embed_size],
                name='embed'
            )

            in_layers_innerprod = tf.reshape(
                tf.matmul(
                    tf.reshape(in_layers_embed, [-1, embed_size]),
                    self.char_embeddings,
                    transpose_b=True
                ),
                [-1, in_size, vocab_size],
                name='innerprod'
            )
            in_layers_distro = tf.nn.softmax(
                in_layers_innerprod,
                axis=2,
                name='distro'
            )

        with tf.variable_scope('recurr'):
            lstm_outputs = append_lstm_stack(self, in_layers_embed, recurr_params, 'lstm')
            lstm_size = lstm_outputs.get_shape().as_list()

        with tf.variable_scope('decode'):
            lstm_to_decode = tf.reshape(lstm_outputs, [-1, lstm_size[-1]])

            lstm_decoded = append_dense(
                self, lstm_to_decode,
                {'num_units': embed_size, 'activation': 'tanh'},
                'dense_0'
            )

            embeds = tf.reshape(lstm_decoded, [-1, in_size, vocab_size], name='embed')

            seq_innerprod = tf.reshape(
                tf.matmul(lstm_decoded, self.char_embeddings, transpose_b=True),
                [-1, in_size, vocab_size],
                name='innerprod'
            )

            seq_distro = tf.nn.softmax(
                seq_innerprod,
                axis=2,
                name='distro'
            )

        with tf.variable_scope('output'):
            #output_int = self.make_onehot_decode_layer(seq_distro[:, -1, ...], probabilistic=False)
            output_int = self.make_onehot_decode_layer(seq_distro[:, -1, ...], probabilistic=True, temp=1.0/50)
            output_str = tf.py_func(_func_int_to_str, [output_int], tf.string, name='string')
            output_str.set_shape([None])

        #
        # Set up loss
        #

        loss = loss_softmax_ce(
            tf.reshape(seq_innerprod[:, :-1, :], [-1, vocab_size]),
            tf.reshape(in_layers_onehot[:, 1:, :], [-1, vocab_size]),
        )

        tb_scalars = {}
        return [in_layer], [output_str], [target_layer], embeds, loss, tb_scalars
