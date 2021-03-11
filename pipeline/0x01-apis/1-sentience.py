#!/usr/bin/env python3
"""home planets of all sentient species"""
import requests


def do_request(url):
    """function that do request
    Args:
    -> url: to process

    Return:
    -> json object
    """
    response = requests.get(url)
    return response.json()


def sentientPlanets():
    """names of the home planets
    of all sentient species.

    Return:
    -> names of the home planets
    """
    names = []
    url = 'https://swapi-api.hbtn.io/api/species/'

    while(url is not None):
        res_j = do_request(url)

        for obj in res_j['results']:
            obj_d = obj['designation']
            obj_c = obj['classification']

            if obj_d == 'sentient' or obj_c == 'sentient':
                if obj['homeworld'] is not None:
                    new_r = do_request(obj['homeworld'])
                    names.append(new_r['name'])

        url = res_j['next']

    return names
