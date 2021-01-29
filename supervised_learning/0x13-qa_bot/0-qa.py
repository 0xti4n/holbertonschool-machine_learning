#!/usr/bin/env python3
"""Question Answering"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """finds a snippet of text within a reference
    document to answer a question"""

    pretrained = 'bert-large-uncased-whole-word-masking-finetuned-squad'
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")
    tokenizer = BertTokenizer.from_pretrained(pretrained)

    question_tokens = tokenizer.tokenize(question)
    ref_tokens = tokenizer.tokenize(reference)

    tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + ref_tokens + ['[SEP]']
    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_word_ids)

    len_ref_tok = (len(ref_tokens) + 1)
    input_type_ids = [0] * (1 + len(question_tokens) + 1) + [1] * len_ref_tok

    input_word_ids, input_mask, input_type_ids = map(lambda t: tf.expand_dims(
        tf.convert_to_tensor(t, dtype=tf.int32), 0),
        (input_word_ids, input_mask, input_type_ids))

    outputs = model([input_word_ids, input_mask, input_type_ids])

    start = tf.argmax(outputs[0][0][1:]) + 1
    end = tf.argmax(outputs[1][0][1:]) + 1

    answer_tokens = tokens[start:end + 1]
    if answer_tokens != []:
        answer = tokenizer.convert_tokens_to_string(answer_tokens)
        return answer
    else:
        return None
