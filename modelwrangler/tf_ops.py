"""Module contains common tensorflow operations"""

import string
from multiprocessing import cpu_count

from unidecode import unidecode

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


def make_data_dict(tf_model, x_data, y_data, is_training=False):
    """Make a dict of data for feed_dict"""

    data_dict = {}

    if y_data is not None:
        data_dict[tf_model.target] = y_data

    if is_training is not None:
        data_dict[tf_model.is_training] = is_training

    if isinstance(x_data, list):
        for input_layer, input_data in zip(tf_model.input, x_data):
            data_dict[input_layer] = input_data
    elif x_data is not None:
        data_dict[tf_model.input] = x_data

    return data_dict

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


#
# Text vectorizing tools
#

class TextProcessor(object):
    """Object that handles mapping characters to onehot embeddings
    and back and forth. Generally uses unicode
    """

    MISSING_CHAR = '?'
    PAD_CHAR = ' '
    DEFAULT_CHARS = string.ascii_letters + string.digits

    def __init__(self, good_chars=None):

        if good_chars is None:
            self.good_chars = self.DEFAULT_CHARS
        else:
            self.good_chars = good_chars
        self.good_chars = unidecode(self.good_chars)

        self.char_to_int = {val: key for key, val in enumerate(self.good_chars)}
        self.int_to_char = {key: val for key, val in enumerate(self.good_chars)}

        self.num_chars = len(self.char_to_int)

        self.missing_char_idx = self.num_chars
        self.pad_char_idx = self.num_chars + 1

        self.char_to_int[unidecode(self.MISSING_CHAR)] = self.missing_char_idx
        self.char_to_int[unidecode(self.PAD_CHAR)] = self.pad_char_idx

        self.int_to_char[self.missing_char_idx] = unidecode(self.MISSING_CHAR)
        self.int_to_char[self.pad_char_idx] = unidecode(self.PAD_CHAR)

    def string_to_ints(self, in_string, pad_len=None):
        """Take a sting, and turn it into a list of integers"""

        char_list = list(unidecode(in_string))

        if pad_len is not None:
            char_list = char_list[:pad_len]

        int_list = [self.char_to_int.get(c, self.missing_char_idx)  for c in char_list]

        char_len = len(char_list)
        if pad_len is not None and char_len < pad_len:
            pad_size = char_len - pad_len
            int_list.extend([self.pad_char_idx] * pad_size)

        return int_list

    def ints_to_string(self, in_ints):
        """Take a list of ints, turn them into a single string"""

        char_list = [self.int_to_char[c] for c in in_ints if c is not self.pad_char_idx]
        out_string = ''.join(char_list)
        return out_string

