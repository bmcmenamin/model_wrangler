"""
Module contains tensorflow model definitions
"""
import os
import logging
import json
import pprint

import tensorflow as tf

import tf_ops as tops
import dataset_managers as dm


def make_dir(path):
    """Initialize directory
    """
    logging.info('Save directory : %s', path)

    try:
        os.makedirs(path)
    except OSError:
        logging.info('Directory %s already exists', path)


class BaseNetworkParams(object):
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
        'path': '',
        'meta_filename': '',
        'tb_log_path': '',
        'batch_size': 256,
        'num_epochs': 3,
        'holdout_prop': 0.1,
        'learning_rate': 0.0001
    }

    # default values for model-specific attributes
    MODEL_SPECIFIC_ATTRIBUTES = {
        'name': 'newmodel',
        'in_size': 10,
        'out_size': 3,
        'max_iter': 500,
    }

    def __str__(self):
        return pprint.pformat(vars(self), indent=4)

    def __repr__(self):
        return self.__str__()

    def __init__(self, kwargs):

        # Set required attributes from kwargs or defaults
        for attr in self.REQUIRED_ATTRIBUTES:
            setattr(self, attr, kwargs.get(attr, self.REQUIRED_ATTRIBUTES[attr]))

        # Set model-specific attributes from kwargs or defaults
        for attr in self.MODEL_SPECIFIC_ATTRIBUTES:
            setattr(self, attr, kwargs.get(attr, self.MODEL_SPECIFIC_ATTRIBUTES[attr]))

        # path is required, but doesn't have a simple default so we specify it
        # here at the time the model is initialized and 'name' has been set
        self.path = kwargs.get('path', os.path.join(os.path.curdir, self.name))
        self.meta_filename = os.path.join(self.path, 'saver-meta')
        self.tb_log_path = os.path.join(self.path, 'tb_log')

        make_dir(self.path)
        make_dir(self.tb_log_path)


    def save(self):
        """save model params to JSON
        """
        make_dir(self.path)

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
        - `setup_training` define training step

    And change the variable `PARAM_CLASS` to point to an approriate

    """

    # there's going to be a lot of attributes in this once
    # we add more layers, so let's ust turn this off now...
    #
    # pylint: disable=too-many-instance-attributes


    PARAM_CLASS = BaseNetworkParams
    DATA_CLASS = dm.DatasetManager

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

        loss = tops.loss_sigmoid_ce(target_layer, in_layer)

        return in_layer, out_layer, target_layer, loss

    def setup_training(self, learning_rate):
        """Set up loss and training step
        """

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        #optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
        #optimizer = tf.train.AdamOptimizer(learning_rate)
        #optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, rho=0.9)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = optimizer.minimize(self.loss)

        return train_step

    def setup_tensorboard_tracking(self, tb_log_path):
        """Set up summary stats to track in tensorboard
        """
        tf.summary.scalar('training_loss', self.loss)
        tb_writer = tf.summary.FileWriter(tb_log_path, self.graph)
        return tb_writer

    def __init__(self, params):
        """Initialize a tensorflow model
        """

        self.graph = tf.Graph()

        with self.graph.as_default():
            self.is_training = tf.placeholder("bool", name="is_training")
            self.input, self.output, self.target, self.loss = self.setup_layers(params)
            self.train_step = self.setup_training(params.learning_rate)

            self.tb_writer = self.setup_tensorboard_tracking(params.tb_log_path)
            self.tb_stats = tf.summary.merge_all()

            self.saver = tf.train.Saver(
                name=params.name,
                filename=params.meta_filename,
                pad_step_number=True,
                max_to_keep=4
                )

