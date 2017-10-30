"""
Module sets up Dense Autoencoder model
"""

import tensorflow as tf

from model_wrangler.model_wrangler import ModelWrangler
import model_wrangler.tf_ops as tops
from model_wrangler.tf_models import BaseNetworkParams, BaseNetwork, LayerConfig    

class DenseFeedforwardParams(BaseNetworkParams):
    """Dense autoencoder params
    """
    MODEL_SPECIFIC_ATTRIBUTES = {
        "name": "ff",
        "in_size": 10,
        "out_size": 2,
        "hidden_nodes": [5, 5],
        "hidden_params": LayerConfig(dropout_rate=0.1),
        "output_params": LayerConfig(
            dropout_rate=None,
            activation=None,
            act_reg=None
        ),
    }


class DenseFeedforwardModel(BaseNetwork):
    """Dense autoencoder model
    """

    # pylint: disable=too-many-instance-attributes

    PARAM_CLASS = DenseFeedforwardParams

    def setup_layers(self, params):
        """Build all the model layers
        """

        #
        # Input layer
        #
        layer_stack = [
            tf.placeholder(
                "float",
                name="input",
                shape=[None, params.in_size]
                )
        ]
        in_layer = layer_stack[0]

        for idx, num_nodes in enumerate(params.hidden_nodes):
            layer_stack.append(
                self.make_dense_layer(
                    layer_stack[-1],
                    num_nodes,
                    'hidden_{}'.format(idx),
                    params.hidden_params
                    )
            )

        out_layer = self.make_dense_layer(
            layer_stack[-1],
            params.out_size,
            'output_layer',
            params.output_params
            )

        target_layer = tf.placeholder(
            "float",
            name="target",
            shape=[None, params.out_size]
        )

        loss = tops.loss_sigmoid_ce(target_layer, out_layer)

        return in_layer, out_layer, target_layer, loss


class DenseFeedforward(ModelWrangler):
    """Dense Autoencoder
    """
    def __init__(self, in_size=10, **kwargs):
        super(DenseFeedforward, self).__init__(
            model_class=DenseFeedforwardModel,
            in_size=in_size,
            **kwargs)
