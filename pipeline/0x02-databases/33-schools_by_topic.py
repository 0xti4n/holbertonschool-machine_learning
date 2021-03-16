#!/usr/bin/env python3
""" Where can I learn Python? """


def schools_by_topic(mongo_collection, topic):
    """returns the list of school having a specific topic

    Args:
    -> mongo_collection will be the pymongo collection object
    -> topic (string) will be topic searched

    Returns:
    -> the specific topic
    """
    obj = mongo_collection.find({'topics': {'$all': [topic]}})

    data = [i for i in obj]
    return data
