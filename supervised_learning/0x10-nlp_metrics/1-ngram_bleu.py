#!/usr/bin/env python3
"""N-gram BLEU score"""
import numpy as np


def n_gram(data, n):
    """creates the n-gram"""

    need_convert = False
    if not isinstance(data[0], list):
        data = [data]
        need_convert = True

    word_list = []
    for line in data:
        new_word = []
        for w_range in range(len(line) - n + 1):
            for i in range(n):
                if i == 0:
                    new_gram = ''
                    new_gram += line[w_range + i]
                else:
                    new_gram += ' '
                    new_gram += line[w_range + i]
            new_word.append(new_gram)
        word_list.append(new_word)

    if need_convert:
        return word_list[0]

    return word_list


def ngram_bleu(references, sentence, n):
    """calculates the n-gram BLEU score for a sentence:

    -> references is a list of reference translations
        * each reference translation is a list
            of the words in the translation

    -> sentence is a list containing the model proposed sentence
    -> n is the size of the n-gram to use for evaluation
    -> Returns: the n-gram BLEU score
    """
    references = n_gram(references, n)
    sentence = n_gram(sentence, n)
    sent_length = len(sentence)
    dict_sentences = {}
    diff_ref_and_sentences = []
    max_dict = {}
    n = n - 1

    for gram in sentence:
        dict_sentences[gram] = dict_sentences.get(gram, 1)

    for ref in references:
        actual_gram = {}
        for gram in ref:
            actual_gram[gram] = actual_gram.get(gram, 1)
        for gram in actual_gram:
            max_dict[gram] = max(max_dict.get(gram, 0), actual_gram[gram])
        diff_ref_and_sentences.append(len(ref) - sent_length)

    counter = 0
    for gram in dict_sentences:
        counter += min(max_dict.get(gram, 0), dict_sentences[gram])

    idx = np.argmin(diff_ref_and_sentences)
    ref_length = len(references[idx])

    if sent_length >= ref_length:
        brevity = 1
    else:
        brevity = np.exp(1 - (ref_length + n) / (sent_length + n))

    result = brevity * counter / sent_length

    return result
