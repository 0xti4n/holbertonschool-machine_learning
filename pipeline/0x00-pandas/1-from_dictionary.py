#!/usr/bin/env python3
"""From Dictionary dataframe"""
import pandas as pd


labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
          'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
          'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

data = {
    'First': [0.0, 0.5, 1.0, 1.5],
    'Second': ['one', 'two', 'three', 'four']
    }

idx = [labels[i].upper() for i in range(len(data['First']))]

df = pd.DataFrame(data, index=idx)
