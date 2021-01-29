#!/usr/bin/env python3
"""Multi-reference Question Answering"""
question_answer = __import__('0-qa').question_answer
semantic_search = __import__('3-semantic_search').semantic_search


def qa_bot(coprus_path):
    """answers questions from multiple reference texts"""

    cases = ['exit', 'quit', 'goodbye', 'bye']
    while(1):
        text = input('Q: ')
        if text.lower() in cases:
            print('A: Goodbye')
            break
        else:
            reference = semantic_search(coprus_path, text)
            answer = question_answer(text, reference)
            if answer is None:
                print('A: Sorry, I do not understand your question.')
            else:
                print('A: {}'.format(answer))
