"""
Module contains tensorflow model definitions
"""
import os
import logging
import json

import tensorflow as tf

import tf_ops as tops


class BaseModelParams(object):
    """
    Parse the model params opts passed in as kwargs.

    You should not implement this class directly and instead define classes
    that inherit from it.

    You'll probably want to redefine the class variable MODEL_SPECIFIC_ATTRIBUTES
    to hold defaul values for your new model
    """

    # all params are stored as attributes, so we need pylint to
    # shut up about having too many instance attributes.
    #
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=E1101


    # default values for required attributes
    REQUIRED_ATTRIBUTES = {
        'verb': True,
        'name': 'newmodel',
        'path': '',
        'batch_size': 256,
        'num_epochs': 3,
    }

    # default values for model-specific attributes
    MODEL_SPECIFIC_ATTRIBUTES = {
        'in_size': 10,
        'out_size': 3,
        'max_iter': 500,
    }


    def __init__(self, kwargs):

        # Set required attributes from kwargs or defaults
        for attr in self.REQUIRED_ATTRIBUTES:
            setattr(self, attr, kwargs.get(attr, self.REQUIRED_ATTRIBUTES[attr]))

        # path is required, but doesn't have a simple default so we specify it
        # here at the time the model is initialized and 'name' has been set
        self.path = kwargs.get('path', os.path.join(os.path.curdir, self.name))

        # Set model-specific attributes from kwargs or defaults
        for attr in self.MODEL_SPECIFIC_ATTRIBUTES:
            setattr(self, attr, kwargs.get(attr, self.MODEL_SPECIFIC_ATTRIBUTES[attr]))


        logging.basicConfig(
            filename=os.path.join(self.path, '{}.log'.format(self.name)),
            level=logging.INFO)


    def init_save_dir(self):
        """Initialize save dir
        """
        logging.info('Save directory : %s', self.path)

        try:
            os.makedirs(self.path)
        except OSError:
            logging.warn('Save directory already exists')

    def find_metagraph(self):
        """Find the most recent meta file with this model
        """
        meta_list = os.path.join(self.path, '*.meta')
        newest_meta_file = max(meta_list, key=os.path.getctime)
        return newest_meta_file

    def save(self):
        """save model params to JSON
        """

        self.init_save_dir()

        params_fname = os.path.join(
            self.path,
            '-'.join([self.name, 'params.json'])
            )
        logging.info('Saving parameter file %s', params_fname)

        with open(params_fname, 'wt') as json_file:
            json.dump(
                vars(self),
                json_file,
                ensure_ascii=True,
                indent=4)


class BaseNetwork(object):
    """
    Base class for tensorflow network. You should not implement this class directly
    and instead define classes that inherit from it.

    Your subclass should redefine the following methods:
        - `setup_layers` should build the whole model
        - `setup_training` define loss function and training step

    And change the variable `PARAM_CLASS` to point to an approriate

    """

    # there's going to be a lot of attributes in this once
    # we add more layers, so let's ust turn this off now...
    #
    # pylint: disable=too-many-instance-attributes


    PARAM_CLASS = BaseModelParams

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

