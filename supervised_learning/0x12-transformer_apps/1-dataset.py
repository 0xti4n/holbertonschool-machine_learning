#!/usr/bin/env python3
"""Encode Tokens"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset():
    """Class Dataset"""
    def __init__(self):
        """INIT class"""
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train',
                                    as_supervised=True)

        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation',
                                    as_supervised=True)

        pt, en = self.tokenize_dataset(self.data_train)

        self.tokenizer_pt = pt
        self.tokenizer_en = en

    def tokenize_dataset(self, data):
        """tokenize data"""
        tokenizator = tfds.features.text.SubwordTextEncoder.build_from_corpus
        tokenizer_en = tokenizator((en.numpy() for _, en in data.repeat(1)),
                                   target_vocab_size=2**15)

        tokenizer_pt = tokenizator((pt.numpy() for pt, _ in data.repeat(1)),
                                   target_vocab_size=2**15)

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """encode data"""
        v_size = [self.tokenizer_pt.vocab_size]
        end_v_size = [self.tokenizer_pt.vocab_size+1]
        pt_tokens = v_size + self.tokenizer_pt.encode(pt.numpy()) + end_v_size

        en_v_size = [self.tokenizer_en.vocab_size]
        end_en_v_size = [self.tokenizer_en.vocab_size+1]
        en_tokens = en_v_size + self.tokenizer_en.encode(en.numpy()) + \
            end_en_v_size

        return pt_tokens, en_tokens
