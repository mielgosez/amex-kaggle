import boto3
import os
import pandas as pd
import pyspark
from pyspark.sql import SparkSession
from amex_pipeline.process_base import BaseETL
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.sql import functions as F
import findspark
findspark.init()
from pyspark.sql.types import *


class ProcessedETL(BaseETL):
    def __init__(self, file_path: str, schema_obj):
        self.__spark_session = SparkSession.builder.master("local[1]").appName("kaggle").config("spark.memory.offHeap.enabled","true").config("spark.memory.offHeap.size","10g").getOrCreate()
        self.__df = self.load_parquet_df(schema_to_load=schema_obj, file_path=file_path)
        self.__target_df = self.session.read.format('csv').option('header', 'true').load('./data/train_labels.csv')

    def load_parquet_df(self, schema_to_load, file_path: str):
        df = self.__spark_session.read.format('parquet').schema(schema_to_load).load(file_path)
        return df

    def apply_one_hot_encoding(self,
                               col: str):
        lcl_fun = F.udf(lambda v, i: float(v[i]), DoubleType())
        idx_col = f'{col}_num'
        cat_col = f'{col}_cat'
        indexer = StringIndexer(inputCol=col, outputCol=idx_col, handleInvalid='keep')
        df_indexed = indexer.fit(self.df).transform(self.df)
        num_elements = df_indexed.select(idx_col).distinct().count()
        encoder = OneHotEncoder(inputCol=f'{col}_num', outputCol=f'{col}_cat')
        df_one_hot = encoder.fit(df_indexed).transform(df_indexed)
        df_one_hot = df_one_hot.drop(col)
        df_one_hot = df_one_hot.drop(idx_col)
        for idx in range(num_elements-1):
            df_one_hot = df_one_hot.withColumn(f'{col}_{idx}', lcl_fun(cat_col, F.lit(idx)))
        df_one_hot = df_one_hot.drop(cat_col)
        self.df = df_one_hot

    def execute(self):
        for col in self.CAT_VAR:
            self.apply_one_hot_encoding(col)
        self.df.drop(self.DATE_COL)
        result = self.df.groupBy(self.CUSTOMER_ID_COL).mean()
        result = result.fillna(0)
        result = result.join(self.df_target, on=self.CUSTOMER_ID_COL, how='left')
        result.write.parquet('./data/preprocessed_model.parquet')
        return result

    @property
    def df(self):
        return self.__df

    @df.setter
    def df(self, new_df: pyspark.sql.DataFrame):
        self.__df = new_df

    @property
    def df_target(self):
        return self.__target_df

    @df_target.setter
    def df_target(self, new_df: pyspark.sql.DataFrame):
        self.__target_df = new_df

    @property
    def session(self):
        return self.__spark_session


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
    local_path = f'./data/processed/df/{id}'
    os.mkdir(local_path)
    # Storing files.
    print(f'Processing {id}')
    for file_obj in objects:
        file_name = file_obj.key.split('/')[-1]
        print('  '+file_name)
        s3_client.download_file(bucket_name, file_obj.key, os.path.join(local_path, file_name))
    return objects
