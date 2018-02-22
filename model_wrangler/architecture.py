"""Module contains tensorflow model definitions"""

import sys
import os
import logging
import json

from abc import ABC, abstractmethod

import tensorflow as tf

import model_wrangler.model.layers as layers
from model_wrangler.model.losses import loss_sigmoid_ce

LOGGER = logging.getLogger(__name__)
h = logging.StreamHandler(sys.stdout)
h.setFormatter(
    logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
)
LOGGER.addHandler(h)
LOGGER.setLevel(logging.DEBUG)


class BaseArchitecture(ABC):
    """
    Base class for tensorflow network. You should not implement this class directly
    and instead define classes that inherit from it.

    Your subclass should redefine the following methods:
        - `setup_layers` should build the whole model
        - `setup_training_step` define training step

    """

    # there's going to be a lot of attributes in this once
    # we add more layers, so let's ust turn this off now...
    #
    # pylint: disable=too-many-instance-attributes

    @abstractmethod
    def setup_layers(self, params):
        """Build all the model layers"""

        # Import params
        in_sizes = params.get('in_sizes', [])
        out_sizes = params.get('out_sizes', [])

        embeds = None

        in_layers = [
            tf.placeholder("float", name="input_{}".format(idx), shape=[None, in_size])
            for idx, in_size in enumerate(in_sizes)
        ]

        out_layers = [
            tf.placeholder("float", name="output_{}".format(idx), shape=[None, out_size])
            for idx, out_size in enumerate(out_sizes)
        ]

        target_layers = [
            tf.placeholder("float", name="target_{}".format(idx), shape=[None, out_size])
            for idx, out_size in enumerate(out_sizes)
        ]

        # sum losses from each output/target pair into a single loss
        loss = tf.reduce_sum(
            [loss_sigmoid_ce(*pair) for pair in zip(target_layers, out_layers)]
        )

        return in_layers, out_layers, target_layers, embeds, loss

    def setup_tensorboard_tracking(self, tb_log_path, name, tensor):
        """Set up summary stats to track in tensorboard"""

        tf.summary.scalar(name, tensor)
        tb_writer = tf.summary.FileWriter(tb_log_path, self.graph)
        return tb_writer

    def setup_training_step(self, params):
        """Set up loss and training step"""

        # Import params
        learning_rate = params.get('learning_rate', 0.01)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = optimizer.minimize(self.loss)

        return train_step

    def __init__(self, params):
        """Initialize a tensorflow model"""

        graph_params = params.get('graph', {})
        train_params = params.get('training', {})

        meta_filename = os.path.join(
            os.path.abspath(params['path']),
            '{}-{}'.format(params['name'], 0)
        )
        print(meta_filename)

        self.inputs = None
        self.outputs = None
        self.targets = None
        self.embeds = None
        self.loss = None
        self.train_step = None

        self.graph = tf.Graph()
        with self.graph.as_default():

            self.is_training = tf.placeholder("bool", name="is_training")

            self.inputs, self.outputs, self.targets, self.embeds, self.loss = self.setup_layers(graph_params)
            self.train_step = self.setup_training_step(train_params)

            self.tb_writer = {
                'training_loss': self.setup_tensorboard_tracking(params['path'], 'training_loss', self.loss),
                'validation_loss': self.setup_tensorboard_tracking(params['path'], 'validation_loss', self.loss),
            }

            self.tb_stats = tf.summary.merge_all()

            self.saver = tf.train.Saver(
                name=params['name'],
                filename=meta_filename,
                pad_step_number=True,
                max_to_keep=4
            )


class BaseTextArchitecture(BaseArchitecture):
    """Architecture set up to do stuff with text"""

    # pylint: disable=too-many-instance-attributes

    def __init__(self, params):
        self.text_map = None
        super().__init__(params)

    def make_onehot_encode_layer(self, in_layer):
        """Return a layer that one-hot encodes an int layer

        Args:
            in_layer: A layer consisting of integer values

        Returns:
            a new layer that has one-hot encoded the integers
        """

        onehot_layer = tf.one_hot(
            tf.to_int32(in_layer),
            len(self.text_map.good_chars) + 2,
            axis=-1
        )

        return onehot_layer


    @staticmethod
    def make_onehot_decode_layer(in_layer, probabilistic=False):
        """Return a layer takes one-hot encoded layer to int
        Args:
            in_layer: A of one-hot endcoded values
            probabilistic: boolean indicating whet er to decode with a weighted
                probabilistic op or the argmax

        Returns:
            a new layer with the index of the largest positive value
        """

        if probabilistic:
            out_layer = tf.multinomial(
                tf.log(tf.nn.softmax(in_layer, axis=-1))
            , 1)

        else:
            out_layer = tf.argmax(in_layer, axis=-1)

        return out_layer
