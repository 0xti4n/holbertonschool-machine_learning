#!/usr/bin/env python3
"""Stacking Bars"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

persons = ['Farrah', 'Fred', 'Felicia']

apples = fruit[0]
bananas = fruit[1]
oranges = fruit[2]
peaches = fruit[3]

name_fruits = {
    'apples': 'red',
    'bananas': 'yellow',
    'oranges': '#ff8000',
    'peaches': '#ffe5b4'
}

width = 0.5
N = np.arange(len(apples))

plt.bar(N, apples, width, color=name_fruits['apples'])
plt.bar(N, bananas, width, color=name_fruits['bananas'], bottom=apples)
plt.bar(N, oranges, width, color=name_fruits['oranges'], bottom=bananas+apples)
plt.bar(N, peaches, width, color=name_fruits['peaches'],
        bottom=oranges+bananas+apples)

plt.xticks(N, persons)
plt.yticks(np.arange(0, 90, 10))

labels = list(sorted(name_fruits.keys()))
handles = [plt.Rectangle((0, 0), 1, 1,
                         color=name_fruits[label]) for label in labels]
plt.legend(handles, labels)


plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')

plt.savefig('img6.png')
