#!/usr/bin/env python3
"""Gradient Plot"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)

x = np.random.randn(2000) * 10
y = np.random.randn(2000) * 10
z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))


f, ax = plt.subplots()
ax.set_xlabel('x coordinate (m)')
ax.set_ylabel('y coordinate (m)')
ax.set_title('Mountain Elevation')

points = ax.scatter(x, y, c=z, s=30.)

cbar = f.colorbar(points)
cbar.set_label('elevation (m)')

plt.savefig('img100.png')
