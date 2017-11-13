"""Module contains common tensorflow operations
"""
from multiprocessing import cpu_count
import tensorflow as tf

#
# Session config functions
#

def set_session_params(cfg_params=None):
    """Set session configuration. Use this to switch between CPU and GPU tasks"""

    if cfg_params is None:
        cfg_params = {}

    return tf.ConfigProto(**cfg_params)

def set_max_threads(sess_cfg, max_threads=None):
    """Set max threads used in session"""

    if max_threads is None:
        max_threads = cpu_count()

    sess_cfg.intra_op_parallelism_threads = max_threads
    sess_cfg.inter_op_parallelism_threads = max_threads
    sess_cfg.allow_soft_placement = True
    return sess_cfg


#
# Loss functions
#

def layer_logits(layer, pad=1.0e-8):
    """convert probabilities to logits"""

    if pad > 0.5 or pad < 0.0:
        raise ValueError(
            'Logit pad should be in interval (0, 0.5),',
            'but somehow you have {}'.format(pad)
            )
    pad_layer = (1 - 2 * pad) * layer + pad
    logit_values = tf.log(pad_layer) - tf.log(1 - pad_layer)
    return logit_values

def loss_mse(observed, actual):
    """Mean squared error loss"""
    numel = tf.reduce_prod(tf.size(observed))
    return tf.reduce_sum(tf.squared_difference(observed, actual)) / tf.cast(numel, tf.float32)


def loss_sigmoid_ce(observed, actual):
    """Calculate sigmoid cross entropy loss"""

    per_sample_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=actual,
        logits=observed
    )
    numel = tf.reduce_prod(tf.size(observed))
    per_batch_loss = tf.reduce_sum(per_sample_loss) / tf.cast(numel, tf.float32)
    return per_batch_loss

def loss_softmax_ce(observed, actual):
    """Calculate softmax cross entropy loss"""

    per_sample_loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=actual,
        logits=observed
    )
    numel = tf.reduce_prod(tf.size(observed))
    per_batch_loss = tf.reduce_sum(per_sample_loss) / tf.cast(numel, tf.float32)
    return per_batch_loss

def accuracy(observed, actual):
    """Accuracy for one-hot encoded categories"""

    is_correct = tf.equal(tf.argmax(observed, axis=1), tf.argmax(actual, axis=1))
    acc = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    return acc

#
# Layer Utils
#

def fit_to_shape(in_layer, target_shape):
    """Use 0-padding and cropping to make a layer conform to the
    size of a different layer.

    This assumes that the first dimension is batch size and doesn't
    need to be changed.
    """
    current_shape = in_layer.get_shape().as_list()

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
