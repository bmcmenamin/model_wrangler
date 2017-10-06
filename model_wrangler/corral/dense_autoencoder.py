"""
Module sets up Dense Autoencoder model
"""

import tensorflow as tf

from model_wrangler.model_wrangler import ModelWrangler
import model_wrangler.tf_ops as tops
from model_wrangler.tf_models import BaseNetworkParams, BaseNetwork, LayerConfig

class DenseAutoencoderParams(BaseNetworkParams):
    """Dense autoencoder params
    """

    MODEL_SPECIFIC_ATTRIBUTES = {
        'name': 'autoenc',
        'in_size': 10,
        'encode_nodes': [5, 5],
        'encode_params': LayerConfig(),
        'decode_nodes': [5, 5],
        'decode_params': LayerConfig(dropout_rate=None),
        'bottleneck_dim': 3,
        'bottleneck_params': LayerConfig(dropout_rate=None),
        'output_params': LayerConfig(activation=None),
    }


class DenseAutoencoderModel(BaseNetwork):
    """Dense autoencoder model
    """

    # pylint: disable=too-many-instance-attributes

    PARAM_CLASS = DenseAutoencoderParams

    def setup_layers(self, params):
        """Build all the model layers
        """

        #
        # Input and encoding layers
        #
        encode_layers = [
            tf.placeholder(
                "float",
                name="input",
                shape=[None, params.in_size]
                )
        ]

        for idx, num_nodes in enumerate(params.encode_nodes):
            encode_layers.append(
                self.make_dense_layer(
                    encode_layers[-1],
                    num_nodes,
                    'encode_{}'.format(idx),
                    params.encode_params
                    )
            )

        #
        # Bottleneck and decoding layers
        #
        decode_layers = [
            self.make_dense_layer(
                encode_layers[-1],
                params.bottleneck_dim,
                'bottleneck',
                params.bottleneck_params
                )
        ]

        for idx, num_nodes in enumerate(params.decode_nodes):
            decode_layers.append(
                self.make_dense_layer(
                    decode_layers[-1],
                    num_nodes,
                    'decode_{}'.format(idx),
                    params.decode_params
                    )
            )

        in_layer = encode_layers[0]
        out_layer = self.make_dense_layer(
            decode_layers[-1],
            params.in_size,
            'output_layer',
            params.decode_params
            )

        target_layer = tf.placeholder(
            "float",
            name="target",
            shape=[None, params.in_size]
        )

        loss = tops.loss_mse(target_layer, out_layer)

        return in_layer, out_layer, target_layer, loss



class DenseAutoencoder(ModelWrangler):
    """Dense Autoencoder
    """

    def __init__(self, in_size=10, **kwargs):
        super(DenseAutoencoder, self).__init__(
            model_class=DenseAutoencoderModel,
            in_size=in_size,
            **kwargs)
