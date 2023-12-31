import boto3
import yaml
from botocore.exceptions import NoCredentialsError
import sys


def download_file_from_s3(bucket_name, file_name, local_path, region, access_key, secret_key):
    s3 = boto3.client('s3', region_name=region, aws_access_key_id=access_key, aws_secret_access_key=secret_key)
    boto3.client()

    try:
        s3.download_file(bucket_name, file_name, local_path)
        print(f"File '{file_name}' downloaded successfully to '{local_path}'.")
    except NoCredentialsError:
        print("Error: AWS credentials not found. Make sure to configure them correctly.")


# Loading the parameters
params_file = sys.argv[1]

with open(params_file, 'r') as fd:
    params = yaml.safe_load(fd)

# Getting all the download parameters
bucket_n = params['download']['bucket_name']
files_to_download = params['download']['bucket_name']
local_destination_path = params['download']['bucket_name']
reg = params['download']['bucket_region']
ak = params['download']['access_key']
sk = params['download']['secret_key']

# Downloading all the files
for file in files_to_download:
    download_file_from_s3(bucket_n, file, local_destination_path + '/' + file, reg, ak, sk)
