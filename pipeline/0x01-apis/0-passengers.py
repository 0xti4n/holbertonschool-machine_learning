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
    try:
        res_j = do_request(url)

        while(True):
            for obj in res_j['results']:
                try:
                    token = obj['passengers'].split(',')
                    if int(token[0]) > passengerCount:
                        starships.append(obj['name'])
                except ValueError:
                    continue

            if res_j['next'] is None:
                return starships

            if res_j['next'] is not None:
                res_j = do_request(res_j['next'])

    except Exception:
        return starships
