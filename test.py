import os
import pandas as pd
from amex_pipeline.data_schema import create_schema
from amex_pipeline.process_raw_data import RawDataETL
from amex_pipeline.schemae.schema_raw import schema_amex


def _test_small_train_data():
    a = pd.read_csv('../data/raw/small_train_data.csv', encoding='utf-16')
    create_schema(a)
    assert True


def test_read_user_data():
    connection = RawDataETL(file_path=os.environ['raw_data_path'],
                            schema_obj=schema_amex)
    connection.execute()
    assert True
