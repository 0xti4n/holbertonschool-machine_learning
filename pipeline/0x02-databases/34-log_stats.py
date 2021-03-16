#!/usr/bin/env python3
"""Log stats"""
from pymongo import MongoClient


if __name__ == "__main__":
    client = MongoClient('mongodb://127.0.0.1:27017')
    nginx_collection = client.logs.nginx

    count_logs = nginx_collection.count_documents({})
    path = {'method': 'GET', 'path': '/status'}
    status = nginx_collection.count_documents(path)
    method = [
        "GET",
        "POST",
        "PUT",
        "PATCH",
        "DELETE"
    ]
    print('{} logs'.format(count_logs))
    print('Methods:')
    for m in method:
        n_data = nginx_collection.count_documents({'method': m})
        print('\tmethod {}: {}'.format(m, n_data))

    print('{} status check'.format(status))
