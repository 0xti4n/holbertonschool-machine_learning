#!/usr/bin/env python3
"""prints the location of a specific user"""
import requests
from sys import argv
import time


if __name__ == '__main__':
    url = argv[1]
    H = {'Accept': 'application/vnd.github.v3+json'}

    response = requests.get(url, params=H)
    res_j = response.json()

    if response.status_code == 404:
        print('Not found')

    elif response.status_code == 403:
        x_limit = response.headers['X-RateLimit-Reset']
        x = (int(x_limit) - int(time.time()) // 60)
        print('Reset in {} min'.format(x))

    else:
        print(res_j['location'])
