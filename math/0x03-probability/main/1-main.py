#!/usr/bin/env python3

import numpy as np
Poisson = __import__('poisson').Poisson

np.random.seed(0)
data = np.random.poisson(5., 100).tolist()
p1 = Poisson(data)
print('P(9):', p1.pmf(9))

p2 = Poisson(lambtha=5)
print('P(9):', p2.pmf(9))

"""
alexa@ubuntu-xenial:0x03-probability$ 
P(9): 0.03175849616802446
P(9): 0.036265577412911795
"""