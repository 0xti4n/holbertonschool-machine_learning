#!/usr/bin/env python3
"""script that displays the upcoming launch"""
import requests
import time


def do_request(url):
    """function that do request
    Args:
    -> url: to process

    Return:
    -> json object
    """
    response = requests.get(url)
    return response.json()


if __name__ == '__main__':
    url = 'https://api.spacexdata.com/v4/launches/upcoming'

    res_j = do_request(url)
    d_unix = int(time.time())

    for i, obj in enumerate(res_j):
        if d_unix > obj['date_unix']:
            d_unix = obj['date_unix']
            idx = i

    url_rocket = 'https://api.spacexdata.com/v4/rockets/'
    url_launch = 'https://api.spacexdata.com/v4/launchpads/'
    rocket_id = res_j[idx]['rocket']
    launch_id = res_j[idx]['launchpad']
    url_rocket = url_rocket + rocket_id
    url_launch = url_launch + launch_id
    data_launch = do_request(url_launch)
    data_rocket = do_request(url_rocket)

    d = {
        'launch_name': res_j[idx]['name'],
        'date': res_j[idx]['date_local'],
        'rkt_name': data_rocket['name'],
        'lpad_name': data_launch['name'],
        'lpad_locality': data_launch['locality'],
    }

    print('{} ({}) {} - {} ({})'.format(d['launch_name'],
                                        d['date'],
                                        d['rkt_name'],
                                        d['lpad_name'],
                                        d['lpad_locality']))
