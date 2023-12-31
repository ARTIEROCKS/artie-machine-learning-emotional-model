#!/usr/bin/python
from pymongo import MongoClient
from base64 import b64decode
import os
import sys
import yaml

# Loading the parameters
params_file = sys.argv[1]

with open(params_file, 'r') as fd:
    params = yaml.safe_load(fd)

# 1- get the connection parameters
serverHost = params['artie_videos']['db']['host']
serverPort = params['artie_videos']['db']['port']
serverUser = params['artie_videos']['db']['user']
serverPassword = params['artie_videos']['db']['password']
serverDb = params['artie_videos']['db']['name']
directory = params['artie_videos']['download_path']

# 2- If the directory does not exists, it creates the directory
if not os.path.exists(directory):
    os.makedirs(directory)

# 3- Connects with the database and the PedagogicalSoftwareData collection
client = MongoClient(
    'mongodb://' + serverUser + ':' + serverPassword + '@' + serverHost + ':' + str(serverPort) + '/' + serverDb)
db = client[serverDb]
sensorData = db.SensorData

# 4- Opens the file to write the database information
for registry in sensorData.find({"sensorObjectType": "VIDEO"}):
    data_uri = registry['data'][1:-1]
    if data_uri != 'data:':
        try:
            header, encoded = data_uri.split(",", 1)
            data = b64decode(encoded)
            file = open(directory + "/" + registry['externalId'] + "_" + str(registry['_id']) + ".webm", 'wb')
            file.write(data)
            file.close()
            print()
        except:
            print("Error")
