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

    if isinstance(tf_model.input, list):
        for input_layer, input_data in zip(tf_model.input, x_data):
            data_dict[input_layer] = input_data
    elif x_data is not None:
        data_dict[tf_model.input] = x_data

    return data_dict


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

        int_list = [self.char_to_int.get(c, self.missing_char_idx) for c in char_list]

        char_len = len(char_list)
        if pad_len is not None and char_len < pad_len:
            pad_size = pad_len - char_len
            int_list.extend([self.pad_char_idx] * pad_size)

        return int_list

    def ints_to_string(self, in_ints):
        """Take a list of ints, turn them into a single string"""

        char_list = [self.int_to_char[c] for c in in_ints if c is not self.pad_char_idx]
        out_string = ''.join(char_list)
        return out_string
