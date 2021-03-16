#!/usr/bin/env python3
""" Insert a document in Python"""


def insert_school(mongo_collection, **kwargs):
    """ insert document pymongo

    Args:
    -> mongo_collection will be the pymongo collection object

    Returns:
    -> the new _id
    """
    obj = mongo_collection.insert_one(kwargs).inserted_id

    return obj
