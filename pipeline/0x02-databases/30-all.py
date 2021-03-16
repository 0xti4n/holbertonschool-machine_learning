#!/usr/bin/env python3
""" List all documents in Python  """


def list_all(mongo_collection):
    """lists all documents in a collection:

    Args:
    -> mongo_collection the collection to list

    ReturnS:
    -> an empty list if no document in the collection
    """
    coll = [i for i in mongo_collection.find()]

    return coll
