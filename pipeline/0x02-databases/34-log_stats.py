#!/usr/bin/env python3
"""Log stats"""
from pymongo import MongoClient


if __name__ == "__main__":
    client = MongoClient('mongodb://127.0.0.1:27017')
    nginx_collection = client.logs.nginx

    count_logs = nginx_collection.count()
    status = nginx_collection.count({'path': '/status'})
    method = {
        "GET": 0,
        "POST": 0,
        "PUT": 0,
        "PATCH": 0,
        "DELETE": 0
    }

    for m in method.keys():
        q = nginx_collection.count({'method': m})
        method[m] = q

    print('{} logs'.format(count_logs))
    print('Methods:')
    print('\t method GET: {}'.format(method['GET']))
    print('\t method POST: {}'.format(method['POST']))
    print('\t method PUT: {}'.format(method['PUT']))
    print('\t method PATCH: {}'.format(method['PATCH']))
    print('\t method DELETE: {}'.format(method['DELETE']))
    print('{} status check'.format(status))
