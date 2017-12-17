"""Module sets up Convolutional Text Classifier"""

import tensorflow as tf

from modelwrangler.model_wrangler import ModelWrangler
import modelwrangler.tf_ops as tops

from modelwrangler.dataset_managers import TextDataManager

from modelwrangler.tf_models import (
    BaseNetworkParams, BaseNetwork,
    ConvLayerConfig, LayerConfig,
)

PAD_LENGTH = 256

class ConvolutionalTextParams(BaseNetworkParams):
    """Convolutional feedforward params
    """

    LAYER_PARAM_TYPES = {
        "conv_params": ConvLayerConfig,
        "dense_params": LayerConfig,
        "output_params": LayerConfig,
    }

    DATASET_MANAGER_PARAMS = {
        "holdout_prop": 0.1,    
        'pad_len': PAD_LENGTH
    }

    MODEL_SPECIFIC_ATTRIBUTES = {
        "name": "conv_text",
        "in_size": PAD_LENGTH,
        "out_size": 1,
        "conv_nodes": [5],
        "conv_params": {
            "dropout_rate": 0.1,
            "kernel": 3,
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


class ConvolutionalTextModel(BaseNetwork):
    """Convolutional feedforward model that has a
    couple convolutional layers and a couple of dense
    layers leading up to an output
    """

    # pylint: disable=too-many-instance-attributes

    PARAM_CLASS = ConvolutionalTextParams
    DATA_CLASS = TextDataManager

    def setup_layers(self, params):
        """Build all the model layers
        """

        #
        # Input layer
        #

        # Input and encoding layers
        layer_stack = [
            tf.placeholder(
                tf.int32,
                name="input",
                shape=[None, params.in_size]
                )
        ]
        in_layer = layer_stack[0]

        good_chars = getattr(params, 'good_chars', tops.TextProcessor.DEFAULT_CHARS)
        character_depth = len(good_chars) + 2

        layer_stack.append(
            self.make_onehot_encode_layer(
                layer_stack[-1],
                character_depth
            )
        )

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
        else:
            loss = tops.loss_softmax_ce(preact_out_layer, target_layer)

        return in_layer, out_layer, target_layer, loss


class ConvolutionalText(ModelWrangler):
    """Dense Autoencoder
    """
    def __init__(self, in_size=10, **kwargs):
        super(ConvolutionalText, self).__init__(
            model_class=ConvolutionalTextModel,
            in_size=in_size,
            **kwargs)
