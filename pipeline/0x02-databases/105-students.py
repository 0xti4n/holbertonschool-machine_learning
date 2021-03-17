#!/usr/bin/env python3
""" Top students with aggregation pipeline mongodb"""


def top_students(mongo_collection):
    """ function that returns all students sorted by average score

    Args:
    -> mongo_collection pymongo collection object

    Retruns:
    -> all students sorted by average score
    """
    key_to_set = 'averageScore'
    p_names = [
        {'$project': {'_id': 1, 'name': 1}}
    ]

    pipeline = [
        {'$unwind': '$topics'},
        {'$group': {'_id': '$_id', key_to_set: {'$avg': '$topics.score'}}},
        {'$sort': {key_to_set: -1}}
    ]

    names = mongo_collection.aggregate(p_names)
    scores = mongo_collection.aggregate(pipeline)

    names = [i for i in names]
    scores = [i for i in scores]

    for n in names:
        for s in scores:
            if n['_id'] == s['_id']:
                s['name'] = n['name']
    return scores
