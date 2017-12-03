"""
Module sets up Dense Autoencoder model
"""

import tensorflow as tf

from modelwrangler.model_wrangler import ModelWrangler
import modelwrangler.tf_ops as tops
from modelwrangler.tf_models import (
    BaseNetworkParams, BaseNetwork,
    ConvLayerConfig, LayerConfig
)

class ConvolutionalFeedforwardParams(BaseNetworkParams):
    """Convolutional feedforward params
    """

    LAYER_PARAM_TYPES = {
        "conv_params": ConvLayerConfig,
        "dense_params": LayerConfig,
        "output_params": LayerConfig,
    }

    MODEL_SPECIFIC_ATTRIBUTES = {
        "name": "conv_ff",
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
            "dropout_rate": 0.1,
            "activation": None,
            "act_reg": None
        },

        "output_params": {
            "dropout_rate": 0.0,
            "activation": None,
            "act_reg": None
        },
    }


class ConvolutionalFeedforwardModel(BaseNetwork):
    """Convolutional feedforward model that has a
    couple convolutional layers and a couple of dense
    layers leading up to an output
    """

    # pylint: disable=too-many-instance-attributes

    PARAM_CLASS = ConvolutionalFeedforwardParams

    def setup_layers(self, params):
        """Build all the model layers
        """

        #
        # Input layer
        #

        # Input and encoding layers
        in_shape = [None]
        if isinstance(params.in_size, (list, tuple)):
            in_shape.extend(params.in_size)
        else:
            in_shape.extend([params.in_size, 1])

        layer_stack = [
            tf.placeholder(
                "float",
                name="input",
                shape=in_shape
                )
        ]
        in_layer = layer_stack[0]

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


        # Output layer is broken into two pieces. The pre/post the application
        # of the activation function. That's because the loss functions
        # for cross-entropy rely on the pre-activation scores in preact

        preact_out_layer, out_layer = self.make_dense_output_layer(
            layer_stack[-1],
            params.out_size,
            params.output_params
        )

        target_layer = tf.placeholder(
            "float",
            name="target",
            shape=[None, params.out_size]
        )

        if params.output_params.activation in ['sigmoid']:
            loss = tops.loss_sigmoid_ce(preact_out_layer, target_layer)
        elif params.output_params.activation in ['softmax']:
            loss = tops.loss_softmax_ce(preact_out_layer, target_layer)
        else:
            loss = tops.loss_mse(target_layer, out_layer)

        return in_layer, out_layer, target_layer, loss


class ConvolutionalFeedforward(ModelWrangler):
    """Dense Autoencoder
    """
    def __init__(self, in_size=10, **kwargs):
        super(ConvolutionalFeedforward, self).__init__(
            model_class=ConvolutionalFeedforwardModel,
            in_size=in_size,
            **kwargs)
