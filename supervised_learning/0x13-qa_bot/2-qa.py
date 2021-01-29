#!/usr/bin/env python3
"""Answer Questions"""
question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """answers questions from a reference text"""

    cases = ['exit', 'quit', 'goodbye', 'bye']
    while(1):
        text = input('Q: ')
        if text.lower() in cases:
            print('A: Goodbye')
            break
        else:
            answer = question_answer(text, reference)
            if answer is None:
                print('A: Sorry, I do not understand your question.')
            else:
                print('A: {}'.format(answer))
