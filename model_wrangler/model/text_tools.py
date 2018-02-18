"""Tools for string processing"""

import string
from unidecode import unidecode


class TextProcessor(object):
    """Object that handles mapping characters to onehot embeddings
    and back and forth. Generally uses unicode
    """

    MISSING_CHAR = '?'
    PAD_CHAR = ' '
    DEFAULT_CHARS = string.ascii_letters + string.digits

    def __init__(self, pad_len=None, good_chars=None):

        if good_chars is None:
            self.good_chars = self.DEFAULT_CHARS
        else:
            self.good_chars = good_chars
        self.good_chars = unidecode(self.good_chars)

        self.char_to_int = {val: key for key, val in enumerate(self.good_chars)}
        self.int_to_char = {key: val for key, val in enumerate(self.good_chars)}

        self.num_chars = len(self.char_to_int)

        self.missing_char_idx = self.num_chars
        self.pad_len = pad_len
        self.pad_char_idx = self.num_chars + 1

        self.char_to_int[unidecode(self.MISSING_CHAR)] = self.missing_char_idx
        self.char_to_int[unidecode(self.PAD_CHAR)] = self.pad_char_idx

        self.int_to_char[self.missing_char_idx] = unidecode(self.MISSING_CHAR)
        self.int_to_char[self.pad_char_idx] = unidecode(self.PAD_CHAR)

    def string_to_ints(self, in_string, use_pad=True):
        """Take a sting, and turn it into a list of integers"""

        char_list = list(unidecode(str(in_string, 'utf-8')))

        if use_pad and self.pad_len is not None:
            char_list = char_list[-self.pad_len:]

        int_list = [self.char_to_int.get(c, self.missing_char_idx) for c in char_list]

        char_len = len(char_list)
        if use_pad and self.pad_len is not None and char_len < self.pad_len:
            pad_size = self.pad_len - char_len
            pad = [self.pad_char_idx] * pad_size
            int_list = pad + int_list

        return int_list

    def ints_to_string(self, in_ints):
        """Take a list of ints, turn them into a single string"""

        char_list = [self.int_to_char.get(c, self.pad_char_idx) for c in in_ints]
        out_string = str(''.join(char_list), 'utf-8')
        return out_string
