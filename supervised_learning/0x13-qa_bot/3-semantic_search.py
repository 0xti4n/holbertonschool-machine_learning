#!/usr/bin/env python3
"""Semantic Search"""
import tensorflow_hub as hub
import os
import numpy as np


def semantic_search(corpus_path, sentence):
    """performs semantic search on a
    corpus of documents"""

    embed = hub.load("https://tfhub.dev/google/universal-"
                     + "sentence-encoder-large/5")
    references = [sentence]

    for filename in os.listdir(corpus_path):
        if not filename.endswith('.md'):
            continue

        data = corpus_path + '/' + filename
        with open(data, 'r', encoding='utf-8') as f:
            references.append(f.read())

    embeddings = embed(references)
    corr = np.inner(embeddings, embeddings)
    idx = np.argmax(corr[0, 1:]) + 1

    return references[idx]
