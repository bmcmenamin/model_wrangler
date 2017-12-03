"""
Module sets up Dense Autoencoder model
"""

import tensorflow as tf

from modelwrangler.model_wrangler import ModelWrangler
from modelwrangler.dataset_managers import SiameseDataManager

import modelwrangler.tf_ops as tops

from modelwrangler.tf_models import (
    BaseNetworkParams, BaseNetwork,
    ConvLayerConfig, LayerConfig
)


class ConvolutionalSiameseParams(BaseNetworkParams):
    """Convolutional feedforward params
    """

    LAYER_PARAM_TYPES = {
        "conv_params": ConvLayerConfig,
        "dense_params": LayerConfig,
        "embed_params": LayerConfig
    }

    MODEL_SPECIFIC_ATTRIBUTES = {
        "name": "conv_siam",
        "in_size": 10,
        "out_size": 2,

        "conv_nodes": [5, 5],
        "conv_params": {
            "dropout_rate": 0.1,
            "kernel": 5,
            "strides": 1,
            "pool_size": 2
        },

        "dense_nodes": [5, 5],
        "dense_params": {
            "dropout_rate": None,
            "activation": 'relu',
            "act_reg": None
        },

        "embed_params": {
            "dropout_rate": None,
            "activation": 'linear',
            "act_reg": None
        },

    }


class ConvolutionalSiameseModel(BaseNetwork):
    """Convolutional feedforward model that has a
    couple convolutional layers and a couple of dense
    layers leading up to an output
    """

    # pylint: disable=too-many-instance-attributes

    PARAM_CLASS = ConvolutionalSiameseParams
    DATA_CLASS = SiameseDataManager

    def build_embedding_model(self, in_layer, params):
        """Build the layers for the embedding model
        that maps a single output to the embedding space
        """

        layer_stack = [in_layer]

        # Add conv layers
        for idx, num_nodes in enumerate(params.conv_nodes):
            layer_stack.append(
                self.make_conv_layer(
                    layer_stack[-1],
                    num_nodes,
                    'conv_{}'.format(idx),
                    params.conv_params
                )
            )

        # Flatten convolutional layers
        layer_stack.append(
            tf.contrib.layers.flatten(
                layer_stack[-1]
            )
        )

        # Add dense layers
        for idx, num_nodes in enumerate(params.dense_nodes):
            layer_stack.append(
                self.make_dense_layer(
                    layer_stack[-1],
                    num_nodes,
                    'dense_{}'.format(idx),
                    params.dense_params
                )
            )

        # Add output layer
        _, out_layer = self.make_dense_output_layer(
            layer_stack[-1],
            params.out_size,
            params.embed_params
        )

        return out_layer


    def setup_layers(self, params):

        # Input and encoding layers
        in_shape = [None]
        if isinstance(params.in_size, (list, tuple)):
            in_shape.extend(params.in_size)
        else:
            in_shape.extend([params.in_size, 1])

        input_0 = tf.placeholder(
            "float",
            name="input_0",
            shape=in_shape
        )

        input_1 = tf.placeholder(
            "float",
            name="input_1",
            shape=in_shape
        )

        with tf.variable_scope("siamese") as scope:

            self.embed_in = input_0
            self.embed_out = self.build_embedding_model(input_0, params)
            scope.reuse_variables()
            _embed_out1 = self.build_embedding_model(input_1, params)



        out_siamese = tf.reduce_sum(
            tf.multiply(self.embed_out, _embed_out1),
            axis=1,
            keep_dims=True)

        target_layer = tf.placeholder(
            "float",
            name="target",
            shape=[None, 1]
        )

        loss = tops.loss_sigmoid_ce(out_siamese, target_layer)

        return [input_0, input_1], out_siamese, target_layer, loss


class ConvolutionalSiamese(ModelWrangler):
    """Dense Autoencoder
    """
    def __init__(self, in_size=10, **kwargs):

        super(ConvolutionalSiamese, self).__init__(
            model_class=ConvolutionalSiameseModel,
            in_size=in_size,
            **kwargs)

        self.embed_in = None
        self.embed_out = None


    def get_embedding_score(self, input_x):
        """Get embedding vectors for a set of inputs"""

        embed_vecs = self.sess.run(
            self.tf_mod.embed_out,
            feed_dict={
                self.tf_mod.embed_in: input_x,
                self.tf_mod.is_training: False        
            }
        )

        return embed_vecs