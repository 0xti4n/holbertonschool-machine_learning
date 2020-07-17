#!/usr/bin/env python3
"""Histogram plot """
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)
x = np.arange(110, step=10)
y = np.arange(30, step=5)

plt.xticks(x)
plt.ylim(0, 30)
plt.hist(student_grades, x, facecolor='blue', alpha=0.6,
         edgecolor='black', linewidth=1.2)

plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.title('Project A')
plt.savefig('img4.png')
