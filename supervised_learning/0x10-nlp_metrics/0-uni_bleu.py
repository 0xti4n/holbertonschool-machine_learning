#!/usr/bin/env python3
"""Unigram BLEU score """
import numpy as np


def uni_bleu(references, sentence):
    """calculates the unigram BLEU score for a sentence

    -> references is a list of reference translations
        * each reference translation is a list of the
            words in the translation

    -> sentence is a list containing the model proposed sentence

    -> Returns: the unigram BLEU score
    """
    sent_length = len(sentence)
    dict_sentences = {}
    diff_ref_and_sentences = []
    max_dict = {}

    for word in sentence:
        dict_sentences[word] = dict_sentences.get(word, 1)

    for ref in references:
        actual_ref = {}
        for word in ref:
            actual_ref[word] = actual_ref.get(word, 1)
        for word in actual_ref:
            max_dict[word] = max(max_dict.get(word, 0), actual_ref[word])
        diff_ref_and_sentences.append(np.abs(len(ref) - sent_length))

    counter = 0
    for word in dict_sentences:
        counter += min(max_dict.get(word, 0), dict_sentences[word])

    idx = np.argmin(diff_ref_and_sentences)
    ref_length = len(references[idx])

    if sent_length > ref_length:
        brevity = 1
    else:
        brevity = np.exp(1 - ref_length / sent_length)

    Score = brevity * counter / sent_length

    if Score > 0.4:
        return round(Score, 7)

    return Score
