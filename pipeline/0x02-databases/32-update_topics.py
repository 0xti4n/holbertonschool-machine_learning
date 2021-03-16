#!/usr/bin/env python3
""" update document in Python"""


def update_topics(mongo_collection, name, topics):
    """changes all topics of a school document based on the name

    Args:
    -> mongo_collection will be the pymongo collection object
    -> name (string) will be the school name to update
    -> topics (list of strings) will be the list of topics
    approached in the school
    """
    query = {'name': name}
    newvalues = {'$set': {'topics': topics}}

    mongo_collection.update(query, newvalues)
