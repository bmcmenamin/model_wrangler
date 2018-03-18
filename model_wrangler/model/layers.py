import tensorflow as tf


CONV_FUNCS = {
    1: tf.layers.conv1d,
    2: tf.layers.conv2d,
    3: tf.layers.conv3d,
}


DECONV_FUNCS = {
    1: tf.layers.conv1d,
    2: tf.layers.conv2d_transpose,
    3: tf.layers.conv3d_transpose,
}


POOL_FUNCS = {
    1: tf.layers.max_pooling1d,
    2: tf.layers.max_pooling2d,
    3: tf.layers.max_pooling3d,
}


UNSTRIDE_FUNCS = {
    1: tf.contrib.keras.layers.UpSampling1D,
    2: tf.contrib.keras.layers.UpSampling2D,
    3: tf.contrib.keras.layers.UpSampling3D,
}


UNPOOL_FUNCS = {
    1: tf.contrib.keras.layers.ZeroPadding1D,
    2: tf.contrib.keras.layers.ZeroPadding2D,
    3: tf.contrib.keras.layers.ZeroPadding3D,
}


def get_param_functions(param_dict):
    """Extract functions for  activation and regularization from
    config params
    """
   
    activation_func = getattr(tf.nn, param_dict.get('activation', ''), None)

    _reg = []
    reg_func = None
    for reg_type, reg_str in param_dict.get('activity_reg', {}).items():
        _reg = getattr(tf.keras.regularizers, reg_type, None)
        if _reg:
            reg_func = _reg(reg_str)
            break

    return activation_func, reg_func


def get_layer_dim(in_layer):
    current_shape = in_layer.get_shape().as_list()
    return len(current_shape) - 1


def append_batchnorm(architecture, input_layer, layer_config, name):
    """Append batch normalization to a layer

    Args:
        architecture: model architecture object
        input_layer: the previous layer that feeds into this
        layer_config: dict of layer params
            None required
        name: layer name

    Returns:
        TF layer with Batch Normalization added
    """
    bn_layer = tf.layers.batch_normalization(
        input_layer,
        training=architecture.is_training,
        name=name
    )
    return bn_layer


def append_dropout(architecture, input_layer, layer_config, name):
    """Append dropout to a layer

    Args:
        architecture: model architecture object
        input_layer: the previous layer that feeds into this
        layer_config: dict of layer params
            'dropout_rate' -> int, default = 0.1
        name: layer name

    Returns:
        TF layer with dropout added
    """
    do_layer = tf.layers.dropout(
        input_layer,
        rate=layer_config.get('dropout_rate', 0.1),
        training=architecture.is_training,
        name=name
    )
    return do_layer


def append_dense(architecture, input_layer, layer_config, name):
    """Add dense connections to a layer

    Args:
        architecture: model architecture object
        input_layer: the previous layer that feeds into this
        layer_config: dict of layer params
            'num_units' -> int or list of <dim> ints, default = 5
            'activation' -> activation function, default = 'None'
            'bias' -> bool for bias on/off, default = 'True'
            'activity_reg' -> dict of regularization:strength pair, default = {}
        name: layer name

    Returns:
        TF layer with dense added
    """

    activation_func, reg_func = get_param_functions(layer_config)

    dense_layer = tf.layers.dense(
        input_layer,
        layer_config.get('num_units', 5),
        activation=activation_func,
        use_bias=layer_config.get('bias', True),
        activity_regularizer=reg_func,
        name=name
    )
    return dense_layer


def append_timedense(architecture, input_layer, layer_config, name):
    """Add time distributed dense connections to a layer

    Args:
        architecture: model architecture object
        input_layer: the previous layer that feeds into this
        layer_config: dict of layer params
            'num_units' -> int or list of <dim> ints, default = 5
            'activation' -> activation function, default = 'None'
            'bias' -> bool for bias on/off, default = 'True'
            'activity_reg' -> dict of regularization:strength pair, default = {}
        name: layer name

    Returns:
        TF layer with dense added
    """

    return None


def append_conv(architecture, input_layer, layer_config, name):
    """Add convolutions to a layer

    Args:
        architecture: model architecture object
        input_layer: the previous layer that feeds into this
        layer_config: dict of layer params
            'activation' -> activation function, default = 'None'
            'num_units' -> int or list of <dim> ints, default = 5
            'kernel' -> int or list of <dim> ints, default = 3
            'strides' -> int or list of <dim> ints, default = 1
        name: layer name

    Returns:
        TF layer with convolutions added
    """

    dim = get_layer_dim(input_layer) - 1
    activation_func, reg_func = get_param_functions(layer_config)

    conv_func = CONV_FUNCS[dim]
    conv_layer = conv_func(
        input_layer,
        layer_config.get('num_units', 5),
        layer_config.get('kernel', 3),
        strides=layer_config.get('strides', 1),
        activation=activation_func,
        activity_regularizer=reg_func,
        padding='same',
        name=name
    )
    return conv_layer


def append_deconv(architecture, input_layer, layer_config, name):
    """Add deconvolutions to a layer

    Args:
        architecture: model architecture object
        input_layer: the previous layer that feeds into this
        layer_config: dict of layer params
            'num_units' -> int or list of <dim> ints, default = 5
            'kernel' -> int or list of <dim> ints, default = 3
            'strides' -> int or list of <dim> ints, default = 1
        name: layer name

    Returns:
        TF layer with convolutions added
    """

    dim = get_layer_dim(input_layer) - 1
    deconv_func = DECONV_FUNCS[dim]

    deconv_layer = deconv_func(
        input_layer,
        layer_config.get('num_units', 5),
        layer_config.get('kernel', 3),
        strides=layer_config.get('strides', 1),
        padding='same',
        name=name
    )
    return deconv_layer


def append_maxpooling(architecture, input_layer, layer_config, name):
    """Add max pooling to a layer

    Args:
        architecture: model architecture object
        input_layer: the previous layer that feeds into this
        layer_config: dict of layer params
            'pool_size' -> int or list of <dim> ints, default = 5
            'strides' -> int or list of <dim> ints, default = 1
            'padding' -> 'same' or 'valid', default = 'valid'
        name: layer name

    Returns:
        TF layer with max pooling added
    """

    dim = get_layer_dim(input_layer) - 1 
    pool_func = POOL_FUNCS[dim]

    pool_layer = pool_func(
        input_layer,
        pool_size=layer_config.get('pool_size', 1),
        strides=layer_config.get('strides', 1),
        padding=layer_config.get('padding', 'valid'),
        name=name
    )
    return pool_layer


def append_unpool(architecture, input_layer, layer_config, name):
    """Add max un-pooling to a layer

    Args:
        architecture: model architecture object
        input_layer: the previous layer that feeds into this
        layer_config: dict of layer params
            'pool_size' -> int or list of <dim> ints, default = 5
        name: layer name

    Returns:
        TF layer with un-pooling added
    """

    dim = get_layer_dim(input_layer) - 1
    unpool_func = UNPOOL_FUNCS[dim]

    pool_size = layer_config.get('pool_size', 1)
    if isinstance(pool_size, int):
        pad_size = pool_size - 1
    else:
        pad_size = [i - 1 for i in pool_size]

    unpool_layer = unpool_func(pad_size, name=name)(input_layer)
    return unpool_layer


def append_unstride(architecture, input_layer, layer_config, name):
    """Add max un-striding to a layer

    Args:
        architecture: model architecture object
        input_layer: the previous layer that feeds into this
        layer_config: dict of layer params
            'strides' -> int or list of <dim> ints, default = 5
        name: layer name

    Returns:
        TF layer with un-stride added
    """

    dim = get_layer_dim(input_layer) - 1   
    unstride_func = UNSTRIDE_FUNCS[dim]

    unstride_layer = unstride_func(
        layer_config.get('strides', 1),
        name=name
    )(input_layer)

    return unstride_layer


def append_onehot_encode(architecture, input_layer, layer_config, name):
    """Add one-hot encodingto a layer

    Args:
        architecture: model architecture object
        input_layer: the previous layer that feeds into this
        layer_config: dict of layer params
            'max_int' -> maximum dimension default = 32
        name: layer name

    Returns:
        TF layer with one-hot encoding
    """

    onehot_layer = tf.one_hot(
        tf.to_int32(in_layer),
        layer_config.get('max_int', 32),
        axis=-1
    )

    return onehot_layer


def append_onehot_decode(architecture, input_layer, layer_config, name):
    """Add one-hot decoding to the layer

    Args:
        architecture: model architecture object
        input_layer: the previous layer that feeds into this
        layer_config: dict of layer params
            none requires
        name: layer name

    Returns:
        TF layer with one-hot decoded
    """

    out_layer = tf.argmax(in_layer, axis=-1)
    return out_layer


def fit_to_shape(architecture, in_layer, layer_config, name):
    """Use 0-padding and cropping to make a layer conform to the
    size of a different layer.

    This assumes that the first dimension is batch size and doesn't
    need to be changed.

    Args:
        architecture: model architecture object
        input_layer: the previous layer that feeds into this
        layer_config: dict of layer params
            'target_shape': list of dimensions
        name: layer name

    Returns:
        TF layer that's been reshaped
    """

    current_shape = in_layer.get_shape().as_list()
    target_shape = layer_config.get('target_shape', current_shape)

    # Padding with zeroes
    pad_params = [[0, 0]]
    for dim in zip(current_shape[1:], target_shape[1:]):
        if dim[0] < dim[1]:
            err_tot = dim[1] - dim[0]
            pad_top = err_tot // 2
            pad_bot = err_tot - pad_top
            pad_params.append([pad_top, pad_bot])
        else:
            pad_params.append([0, 0])
    in_layer_padded = tf.pad(in_layer, tf.constant(pad_params), 'CONSTANT')

    # Cropping using slice
    padded_shape = in_layer_padded.get_shape().as_list()

    slice_offsets = [0]
    slice_widths = [-1]
    for dim in zip(padded_shape[1:], target_shape[1:]):
        if dim[0] > dim[1]:
            slice_offsets.append((dim[0] - dim[1]) // 2)
        else:
            slice_offsets.append(0)
        slice_widths.append(dim[1])

    in_layer_padded_trimmed = tf.slice(in_layer_padded, slice_offsets, slice_widths)

    return in_layer_padded_trimmed


def append_bidir_lstm_stack(architecture, input_layer, layer_configs, name):
    """Adds stacked RNN layers"""

    cells_fw = [
        tf.contrib.rnn.DropoutWrapper(
            tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(layer_param['units']),
            input_keep_prob=1 - layer_param.get('dropout', 0.0)
        )
        for idx, layer_param in enumerate(layer_configs)
    ]

    cells_bw = [
        tf.contrib.rnn.DropoutWrapper(
            tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(layer_param['units']),
            input_keep_prob=1 - layer_param.get('dropout', 0.0)
        )
        for idx, layer_param in enumerate(layer_configs)
    ]

    output_sequence, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
        cells_fw=tf.nn.rnn_cell.MultiRNNCell(cells_fw),
        cells_bw=tf.nn.rnn_cell.MultiRNNCell(cells_bw),
        inputs=input_layer,
        dtype=tf.float32
    )

    return output_sequence


def append_lstm_stack(architecture, input_layer, layer_configs, name):
    """Adds stacked RNN layers"""

    cells = [
        tf.contrib.rnn.DropoutWrapper(
            #tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(layer_param['units']),
            tf.contrib.rnn.LSTMBlockCell(layer_param['units']),
            input_keep_prob=1 - layer_param.get('dropout', 0.0)
        )
        for idx, layer_param in enumerate(layer_configs)
    ]

    outputs, _ = tf.nn.dynamic_rnn(
        cell=tf.nn.rnn_cell.MultiRNNCell(cells),
        inputs=input_layer,
        dtype=tf.float32,
        time_major=False
    )

    #output = outputs[:, -1, ...]

    return outputs
