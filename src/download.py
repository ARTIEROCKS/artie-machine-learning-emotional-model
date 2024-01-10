import boto3
import yaml
from botocore.exceptions import NoCredentialsError
import sys
import zipfile
import os


def download_file_from_s3(bucket_name, file_name, local_path, region, access_key, secret_key):
    print(f"Access Key ID '{access_key}' AWS Secret '{secret_key}' FILE NAME '{file_name}' LOCAL PATH '{local_path}'.")
    s3 = boto3.client('s3', region_name=region, aws_access_key_id=access_key, aws_secret_access_key=secret_key)

    try:
        s3.download_file(bucket_name, file_name, local_path)
    except NoCredentialsError:
        print("Error: AWS credentials not found. Make sure to configure them correctly.")


def unzip_file(zip_file, destination_directory):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        # Extract all files to the destination directory
        zip_ref.extractall(destination_directory)
        print(f'Files extracted to: {destination_directory}')

    if os.path.exists(zip_file):
        # Removes zip file
        os.remove(zip_file)


# Loading the parameters
params_file = sys.argv[1]

with open(params_file, 'r') as fd:
    params = yaml.safe_load(fd)

# Getting all the download parameters
bucket_n = params['download']['bucket_name']
files_to_download = params['download']['files_to_download']
local_destination_path = params['download']['local_destination_path']
reg = params['download']['bucket_region']
ak = params['download']['access_key']
sk = params['download']['secret_key']

# Downloading all the files
for file in files_to_download:
    download_file_from_s3(bucket_n, file, str(local_destination_path) + '/' + str(file), reg, ak, sk)
    unzip_file(str(local_destination_path) + '/' + str(file), str(local_destination_path))
