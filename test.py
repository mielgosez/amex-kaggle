import os
import pandas as pd
from amex_pipeline.data_schema import create_schema
from amex_pipeline.process_raw_data import RawDataETL
from amex_pipeline.schemae.schema_raw import schema_amex
from amex_pipeline.process_parquet_raw_data import download_file_from_s3, ProcessedETL
"""
connection = RawDataETL(file_path='./data/raw/small_train_data.csv',
                        schema_obj=schema_amex)
connection.execute()
"""


def _test_small_train_data():
    a = pd.read_csv('./data/raw/small_train_data.csv', encoding='utf-16')
    create_schema(a)
    assert True


def _test_read_user_data():
    connection = RawDataETL(file_path=os.environ['raw_data_path'],
                            schema_obj=schema_amex)
    connection.execute()
    assert True


def _test_load_parquet():
    etl = ProcessedETL(file_path='./data/processed/df/customer_ID',
                       schema_obj=schema_amex)
    etl.execute()


def test_download_s3():
    object_ids = pd.read_csv('./data/train_labels.csv')['customer_ID']
    not_in_s3 = list()
    for num, id_obj in enumerate(object_ids):
        if num % 1000 == 0:
            print(f'id {num}th')
        is_empty = download_file_from_s3(bucket_name=os.environ['bucket_name'],
                                         id=id_obj,
                                         directory_prefix='processed')
        if is_empty:
            not_in_s3.append(id_obj)
    to_csv = pd.DataFrame({'id': not_in_s3})
    to_csv.to_csv('not_in_s3.csv', index=False)
