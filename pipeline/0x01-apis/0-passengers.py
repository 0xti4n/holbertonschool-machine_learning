#!/usr/bin/env python3
"""swap api passangers"""
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


def availableShips(passengerCount):
    """using the Swapi API, create a method that returns the
    list of ships that can hold a given number of passengers

    Args:
    -> passengerCount: number of passanger

    Returns:
    -> list of number of pasangers
    """
    starships = []
    url = 'https://swapi-api.hbtn.io/api/starships/'

    while(url is not None):
        res_j = do_request(url)

        for obj in res_j['results']:
            token = obj['passengers'].replace(',', '')
            if token.isnumeric() and int(token) >= passengerCount:
                starships.append(obj['name'])

        url = res_j['next']

    return starships
