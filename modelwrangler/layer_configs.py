"""Module defines objects used for building Layer configurations"""

import tensorflow as tf

class LayerConfig(object):
    """Make an object that stores layer parameters for easy access using dot notation"""

    # pylint: disable=too-many-arguments

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
        """Return function for layer activation"""
        if self.activation:
            return getattr(tf.nn, self.activation, None)
        return None

    def regularization_func(self):
        """Return function for activity regularization"""
        if self.act_reg:
            reg_list = []
            for reg_type in self.act_reg:
                reg_func = getattr(tf.keras.regularizers, reg_type, None)
                if reg_func:
                    reg_list.append(reg_func(self.act_reg[reg_type]))

            if len(reg_list) > 1:
                raise ValueError('Too many regularization types specified')
            return reg_list[0]
        return None


class ConvLayerConfig(LayerConfig):
    """Make an object that stores layer parameters for a convonulational layer"""

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
        """Return which convolution method to use"""

        if self.dim == 1:
            return tf.layers.conv1d
        elif self.dim == 2:
            return tf.layers.conv2d
        elif self.dim == 3:
            return tf.layers.conv3d

    def deconv_func(self):
        """Return which deconvolution method to use"""

        if self.dim == 1:
            return tf.layers.conv1d
        elif self.dim == 2:
            return tf.layers.conv2d_transpose
        elif self.dim == 3:
            return tf.layers.conv3d_transpose

    def pool_func(self):
        """Return which maxpooling method to use"""

        if self.dim == 1:
            return tf.layers.max_pooling1d
        elif self.dim == 2:
            return tf.layers.max_pooling2d
        elif self.dim == 3:
            return tf.layers.max_pooling3d

    def unstride_func(self):
        """Return which unstride method to use"""

        if self.dim == 1:
            return tf.contrib.keras.layers.UpSampling1D
        elif self.dim == 2:
            return tf.contrib.keras.layers.UpSampling2D
        elif self.dim == 3:
            return tf.contrib.keras.layers.UpSampling3D

    def unpool_func(self):
        """Return which unpool method to use"""

        if self.dim == 1:
            return tf.contrib.keras.layers.ZeroPadding1D
        elif self.dim == 2:
            return tf.contrib.keras.layers.ZeroPadding2D
        elif self.dim == 3:
            return tf.contrib.keras.layers.ZeroPadding3D
