"""
Module contains tensorflow model definitions
"""
import os
import logging
import json

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


class BaseNetworkParams(dict):
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
        "verb": True,
        "path": "",
        "meta_filename": "",
        "tb_log_path": "",
        "batch_size": 256,
        "num_epochs": 3,
        "holdout_prop": 0.1,
        "learning_rate": 0.0001
    }

    # default values for model-specific attributes
    MODEL_SPECIFIC_ATTRIBUTES = {
        "name": "newmodel",
        "in_size": 10,
        "out_size": 3,
        "max_iter": 500,
    }

    def __init__(self, **kwargs):

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

        params_fname = os.path.join(self.path, '-'.join([self.name, 'params.json']))
        logging.info('Saving parameter file %s', params_fname)

        dict_to_dump = vars(self)
        for key in dict_to_dump:
            if issubclass(dict_to_dump[key].__class__, LayerConfig):
                dict_to_dump[key] = dict_to_dump[key].__dict__

        with open(params_fname, 'wt') as json_file:
            json.dump(dict_to_dump, json_file, indent=4)


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

    def _make_batchnorm(self, input_layer, name):
        """Wrap batchnormalization around a layer
        """
        bn_layer = tf.layers.batch_normalization(
            input_layer,
            training=self.is_training,
            name=name
        )
        return bn_layer

    def _make_dropout(self, input_layer, name, layer_config):
        """Wrap dropout layer around a layer
        """
        do_layer = tf.layers.dropout(
            input_layer,
            rate=layer_config.dropout_rate,
            training=self.is_training,
            name=name
        )
        return do_layer

    def _make_conv(self, input_layer, num_units, name, layer_config):
        """Make convolution layer
        """
        conv_layer = layer_config.conv_func()(
            input_layer,
            num_units,
            layer_config.kernel,
            strides=layer_config.strides,
            padding='same',
            name=name
        )
        return conv_layer

    def _make_deconv(self, input_layer, num_units, name, layer_config):
        """Make deconvolution layer
        """
        deconv_layer = layer_config.deconv_func()(
            input_layer,
            num_units,
            layer_config.kernel,
            strides=layer_config.strides,
            padding='same',
            name=name
        )
        return deconv_layer

    def _make_maxpooling(self, input_layer, name, layer_config):
        """Apply max pooling to a layer
        """
        pool_layer = layer_config.pool_func()(
            input_layer,
            pool_size=layer_config.pool_size,
            strides=1,
            name=name
        )
        return pool_layer

    def make_dense_layer(self, input_layer, num_units, label, layer_config):
        """ Make a dense network layer

        activation function/actiation-regularization
        THEN (optional) batch normalization
        THEN (optional) dropout

        """

        if isinstance(layer_config, dict):
            layer_config = LayerConfig(**layer_config)

        assert isinstance(layer_config, LayerConfig)

        name_stack = [label]
        layer_stack = [
            tf.layers.dense(
                input_layer,
                num_units,
                activation=layer_config.activation_func(),
                use_bias=layer_config.bias,
                activity_regularizer=layer_config.act_reg,
                name='_'.join(name_stack)
            )
        ]

        # adding batch normalization
        if layer_config.batchnorm:
            name_stack.append('batchnorm')
            layer_stack.append(
                self._make_batchnorm(layer_stack[-1], '_'.join(name_stack))
            )

        # adding dropout
        if layer_config.dropout_rate:
            name_stack.append('dropout')
            layer_stack.append(
                self._make_dropout(layer_stack[-1], '_'.join(name_stack), layer_config)
            )

        return layer_stack[-1]

    def make_conv_layer(self, input_layer, num_units, label, layer_config):
        """ Make a convolutional network layer

        activation function/actiation-regularization
        THEN (optional) batch normalization
        THEN (optional) pooling
        THEN (optional) dropout
        """

        if isinstance(layer_config, dict):
            layer_config = ConvLayerConfig(**layer_config)

        assert isinstance(layer_config, ConvLayerConfig)

        name_stack = [label]
        layer_stack = [
            self._make_conv(
                input_layer, num_units, '_'.join(name_stack), layer_config)
        ]

        # adding batch normalization
        if layer_config.batchnorm:
            name_stack.append('batchnorm')
            layer_stack.append(
                self._make_batchnorm(
                    layer_stack[-1], '_'.join(name_stack))
            )

        # adding pooling
        if layer_config.pool_size:
            name_stack.append('pooling')
            layer_stack.append(
                self._make_maxpooling(
                    layer_stack[-1], '_'.join(name_stack), layer_config)
            )

        # adding dropout
        if layer_config.dropout_rate:
            name_stack.append('dropout')
            layer_stack.append(
                self._make_dropout(
                    layer_stack[-1], '_'.join(name_stack), layer_config)
            )

        return layer_stack[-1]

    def make_deconv_layer(self, input_layer, num_units, label, layer_config):
        """ Make a convolutional network layer

        activation function/actiation-regularization
        THEN (optional) batch normalization
        THEN (optional) dropout

        """

        if isinstance(layer_config, dict):
            layer_config = ConvLayerConfig(**layer_config)

        assert isinstance(layer_config, ConvLayerConfig)

        name_stack = [label]
        layer_stack = [
            self._make_deconv(
                input_layer, num_units, '_'.join(name_stack), layer_config)
        ]

        # adding batch normalization
        if layer_config.batchnorm:
            name_stack.append('batchnorm')
            layer_stack.append(
                self._make_batchnorm(
                    layer_stack[-1], '_'.join(name_stack))
            )

        # adding dropout
        if layer_config.dropout_rate:
            name_stack.append('dropout')
            layer_stack.append(
                self._make_dropout(
                    layer_stack[-1], '_'.join(name_stack), layer_config)
            )

        return layer_stack[-1]

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

class LayerConfig(object):
    """Make an object that stores layer parameters for easy access using dot notation
    """
    def __str__(self):
        return str(vars(self))

    def __repr__(self):
        return str(vars(self))

    def __init__(
            self, activation='relu', batchnorm=True,
            act_reg=None, dropout_rate=0.1, bias=True,
            **layer_kws):

        self.activation = str(activation)
        self.batchnorm = batchnorm

        self.act_reg = act_reg
        self.bias = bias
        self.layer_kws = layer_kws

        if dropout_rate:
            if dropout_rate >= 1.0 or dropout_rate < 0.0:
                raise ValueError('dropout rate must be between 0 and 1')
        self.dropout_rate = dropout_rate
        
    def activation_func(self):
        return getattr(tf.nn, self.activation, None)

class ConvLayerConfig(LayerConfig):
    """Make an object that stores layer parameters for a convonulational layer
    """
    def __init__(self, kernel=(5, 5), strides=(1, 1), pool_size=(3, 3), **param_dict):
        super(ConvLayerConfig, self).__init__(**param_dict)

        if isinstance(kernel, (list, tuple)):
            self.dim = len(kernel)
        elif isinstance(strides, (list, tuple)):
            self.dim = len(strides)
        elif isinstance(pool_size, (list, tuple)):
            self.dim = len(pool_size)
        else:
            self.dim = 1

        self.kernel = kernel
        self.strides = strides
        self.pool_size = pool_size

    def conv_func(self):
        """Return which convolution method to use
        """
        if self.dim == 1:
            return tf.layers.conv1d
        elif self.dim == 2:
            return tf.layers.conv2d
        elif self.dim == 3:
            return tf.layers.conv3d

    def deconv_func(self):
        """Return which deconvolution method to use
        """
        if self.dim == 1:
            return tf.layers.conv1d
        elif self.dim == 2:
            return tf.layers.conv2d_transpose
        elif self.dim == 3:
            return tf.layers.conv3d_transpose

    def pool_func(self):
        """Return which maxpooling method to use
        """
        if self.dim == 1:
            return tf.layers.MaxPooling1D
        elif self.dim == 2:
            return tf.layers.MaxPooling2D
        elif self.dim == 3:
            return tf.layers.MaxPooling3D
