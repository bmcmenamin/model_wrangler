"""
Module contains tensorflow model definitions
"""

import tensorflow as tf

import tf_ops as tops

class BaseNetwork(object):
    """
    Base class for tensorflow network. You should not implement this class directly
    and instead define classes that inherit from it.

    Your subclass should redefine the following methods:

        - `setup_layers` should build the whole model
        - `setup_training` define loss function and training step
    """

    # there's going to be a lot of attributes in this once
    # we add more layers, so let's ust turn this off now...
    #
    # pylint: disable=too-many-instance-attributes


    def setup_layers(self, params):
        """Build all the model layers
        """
        in_layer = tf.placeholder(
            "float",
            name="input",
            shape=[None, params.in_size]
            )

        out_layer = tf.layers.dense(
            self.input,
            params.out_size,
            name="output"
            )

        target_layer = tf.placeholder(
            "float",
            name="target",
            shape=[None, params.out_size]
            )

        return in_layer, out_layer, target_layer

    def setup_training(self):
        """Set up loss and training step
        """

        loss = tops.loss_sigmoid_ce(self.target, self.output)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        optimizer = tf.train.AdamOptimizer(epsilon=1.0e-4)
        with tf.control_dependencies(update_ops):
            train_step = optimizer.minimize(self.loss)

        return loss, train_step

    def initialize_weights(self):
        """Initialize model weights"""

        initializer = tf.variables_initializer(
            self.graph.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES
                )
            )

        with tf.Session(graph=self.graph) as sess:
            sess.run(initializer)

    def __init__(self, params):
        """Initialize a tensorflow model
        """

        self.graph = tf.Graph()

        with self.graph.as_default():
            self.is_training = tf.placeholder("bool", name="is_training")
            self.input, self.output, self.target = self.setup_layers(params)
            self.loss, self.train_step = self.setup_training()

            self.saver = tf.train.Saver(
                name=params.name,
                pad_step_number=True,
                max_to_keep=4
                )

        self.initialize_weights()

