#!/usr/bin/env python3
"""script that takes in input from the user"""

cases = ['exit', 'quit', 'goodbye', 'bye']
while(1):
    text = input('Q: ')
    if text.lower() in cases:
        print('A: Goodbye')
        break
    else:
        print('A: ')
