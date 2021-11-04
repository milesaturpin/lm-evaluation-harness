import logging
import os
import re
import json

import numpy as np
import regex as re
import tensorflow as tf
import tensorflow_datasets as tfds

class SpecialTokens:

    def __init__(self, vocab_size):
        self.MASK_TOKEN = vocab_size + 1

    def __len__(self):
        return len(self.__dict__.keys())

    def decode(self, i):
        return '#ST' + str(i)

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class OpenWebTextEncoder:

    def __init__(self, hparams=None, **kwargs):

        if kwargs.get("encoder_dir"):
            self.encoder_dir = kwargs['encoder_dir']
        elif hparams.encoder_dir:
            self.encoder_dir = hparams.encoder_dir
        else:
            logging.info(f"No encoder found! Set hparams.encoder_dir in the HParams.")

        self.name = os.path.basename(os.path.normpath(self.encoder_dir))
        logging.info(f"Using encoder {self.name}")

        # Check that the given encoder exists
        encoder_path = os.path.join(self.encoder_dir, 'encoder.json')
        if not tf.io.gfile.exists(encoder_path):
            raise FileNotFoundError(f"Encoder not found at {encoder_path}!")

        self.bpe_data = tf.io.gfile.GFile(os.path.join(self.encoder_dir, 'vocab.bpe'), mode='r').read()

        self.encoder = json.loads(tf.io.gfile.GFile(encoder_path).read())
        special_tokens = SpecialTokens(len(self.encoder))
        self.encoder.update(special_tokens.__dict__)
        self.decoder = {v: k for k, v in self.encoder.items()}

        if hparams:
            hparams.vocab_size = len(self.encoder)

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        bpe_merges = [tuple(merge_str.split()[0:2]) for merge_str in self.bpe_data.split('\n')[1:-1]]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))

        self.cache = {}
        # TODO (nick) make this not a regex since regex is slow
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        self.cls_sep_tokens = self.get_cls_sep_tokens()

    def get_cls_sep_tokens(self):
        """Uses least common tokens as CLS and SEP tokens, for BERT training without
        having to use actual SEP and CLS tokens. This works because bpe_data is ordered from
        most common to least common."""
        ordered_bpe = self.bpe_data.splitlines()
        cls_token_list = ordered_bpe[-1].split(" ")
        cls_token = cls_token_list[0] + cls_token_list[1]
        cls_token = cls_token.replace("Ġ", " ")
        sep_token_list = ordered_bpe[-2].split(" ")
        sep_token = sep_token_list[0] + sep_token_list[1]
        sep_token = sep_token.replace("Ġ", " ")
        return self.encode(cls_token), self.encode(sep_token)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        # Encoding is invertible because all out-of-vocab word-pieces are byte-encoded.
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, ids):

        if not isinstance(ids, (tf.Tensor, np.ndarray, list)):  # model output is a single token
            ids = np.expand_dims(np.int32(ids), axis=0)
        elif isinstance(ids, tf.Tensor):
            ids = ids.numpy()

        tokens = []
        for token in ids:
            # to nicely decode padding and BERT's masking token
            if token != 0 and token != len(self.encoder) + 1:
                tokens.append(self.decoder[token])
            elif token == len(self.encoder) + 1:
                tokens.append('[MASK]')

        text = "".join(tokens)

        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors='replace')
        return text