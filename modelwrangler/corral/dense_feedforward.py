"""Module sets up Dense Autoencoder model"""

import tensorflow as tf

from modelwrangler.model_wrangler import ModelWrangler
import modelwrangler.tf_ops as tops
from modelwrangler.tf_models import BaseNetworkParams, BaseNetwork, LayerConfig    

class DenseFeedforwardParams(BaseNetworkParams):
    """Dense autoencoder params
    """

    LAYER_PARAM_TYPES = {
        "hidden_params": LayerConfig,
        "output_params": LayerConfig,
    }

    MODEL_SPECIFIC_ATTRIBUTES = {
        "name": "ff",
        "in_size": 10,
        "out_size": 2,
        "hidden_nodes": [5, 5],
        "hidden_params": {
            "dropout_rate": 0.1
        },
        "output_params": {
            "dropout_rate": None,
            "activation": None,
            "act_reg": None
        },
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


class DenseFeedforward(ModelWrangler):
    """Dense Autoencoder
    """
    def __init__(self, in_size=10, **kwargs):
        super(DenseFeedforward, self).__init__(
            model_class=DenseFeedforwardModel,
            in_size=in_size,
            **kwargs)
