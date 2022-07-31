import os
import pandas as pd
from amex_pipeline.data_schema import create_schema
from amex_pipeline.process_raw_data import RawDataETL
from amex_pipeline.schemae.schema_raw import schema_amex
from amex_pipeline.process_parquet_raw_data import download_file_from_s3
"""
connection = RawDataETL(file_path='./data/raw/small_train_data.csv',
                        schema_obj=schema_amex)
connection.execute()
"""



def _test_small_train_data():
    a = pd.read_csv('../data/raw/small_train_data.csv', encoding='utf-16')
    create_schema(a)
    assert True


def _test_read_user_data():
    connection = RawDataETL(file_path=os.environ['raw_data_path'],
                            schema_obj=schema_amex)
    connection.execute()
    assert True


def test_download_s3():
    object_ids = ['0000f99513770170a1aba690daeeb8a96da4a39f11fc27da5c30a79db61c1e85',
                  '00013181a0c5fc8f1ea38cd2b90fe8ad2fa8cad9d9f13e4063bdf6b0f7d51eb6',
                  '0001337ded4e1c2539d1a78ff44a457bd4a95caa55ba1730b2849b92ea687f9e',
                  '00013c6e1cec7c21bede7cb319f1e28eb994f5625257f479c53ad6e90c177f7c',
                  '0001812036f1558332e5c0880ecbad70b13a6f28ab04a8db6d83a26ef40aadb0',
                  '0002e335892f7998f0feb3a59f32d652f0da7c85e535b99ea6f87fd317ed47f4',
                  '000333075fb8ec6d504539852eeeb762643562e701ac79b2101ab0f9471eeb5a',
                  '000391f219520dbca6c3c1c46e0fab569da163f79ee266b2cc95fb31029ce617',
                  '00039533fe0b61bcf1ec0d1aefe6acb5469ea0f0d1b0ad59ae721e5b86db12f1',
                  '000473eb907b57c8c23f652bba40f87fe7261273dda47034d46fc46821017e50',
                  '0000099d6bd597052cdcda90ffabf56573fe9d7c79be5fbac11a8ed792feb62a',
                  '00000fd6641609c6ece5454664794f0340ad84dddce9a267a310b5ae68e9d8e5',
                  '00001b22f846c82c51f6e3958ccd81970162bae8b007e80662ef27519fcc18c1',
                  '000041bdba6ecadd89a52d11886e8eaaec9325906c9723355abb5ca523658edc',
                  '00007889e4fcd2614b6cbe7f8f3d2e5c728eca32d9eb8ad51ca8b8c4a24cefed',
                  '000084e5023181993c2e1b665ac88dbb1ce9ef621ec5370150fc2f8bdca6202c',
                  '000098081fde4fd64bc4d503a5d6f86a0aedc425c96f5235f98b0f47c9d7d8d4',
                  '0000d17a1447b25a01e42e1ac56b091bb7cbb06317be4cb59b50fec59e0b6381',
                  '00018dd4932409baf6083519b52113c2ef58be59e1213e4681d28c7719a65ddf',
                  '000198b3dc70edd65dbf0d7eddbcb926c6d7dbd7986af19d91ef3992ae3ab896'
                  ]
    for id_obj in object_ids:
        download_file_from_s3(bucket_name=os.environ['bucket_name'],
                              id=id_obj,
                              directory_prefix='processed')
