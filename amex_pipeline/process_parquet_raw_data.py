import boto3
import os
import pandas as pd


def download_file_from_s3(bucket_name: str,
                          id: str,
                          directory_prefix: str):
    """
    lists files located in s3://bucket_name/directory_prefix/id/ and copy them in
    :param bucket_name:
    :param id:
    :param directory_prefix:
    :return:
    """
    # S3 setup
    s3 = boto3.resource('s3')
    s3_client = boto3.client('s3')
    folder_path = '/'.join([directory_prefix, 'customer_ID='+id])
    bucket = s3.Bucket(bucket_name)
    objects = bucket.objects.filter(Prefix=folder_path).all()
    # Getting local path and creating local folder.
    local_path = f'../data/processed/df/{id}'
    os.mkdir(local_path)
    # Storing files.
    print(f'Processing {id}')
    for file_obj in objects:
        file_name = file_obj.key.split('/')[-1]
        print('  '+file_name)
        s3_client.download_file(bucket_name, file_obj.key, os.path.join(local_path, file_name))
    return objects
