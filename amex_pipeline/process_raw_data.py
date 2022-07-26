import os
import pyspark
from logging import getLogger, INFO, WARNING, ERROR, DEBUG
from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.ml.feature import OneHotEncoder, StringIndexer

default_logger = getLogger('default')


class RawDataETL:
    CAT_VAR = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120',
               'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
    CUSTOMER_ID_COL = 'customer_ID'
    DATE_COL = 'S_2'
    
    def __init__(self, file_path: str, schema_obj, logger_level: int = INFO):
        self.__spark_session = SparkSession.builder.master("local[1]").appName("kaggle").getOrCreate()
        self.__df = self.read_user_data(file_path=file_path, schema_obj=schema_obj)
        default_logger.setLevel(logger_level)
        self.__logger = default_logger

    def execute(self, out_path: str = '../data/processed/df'):
        for col in self.CAT_VAR:
            self.apply_one_hot_encoding(col=col)
        self.replace_id_by_int(col=self.CUSTOMER_ID_COL, new_col='id')
        self.replace_id_by_int(col=self.DATE_COL, new_col='date_num')
        self.df.write.parquet(path=out_path, mode='overwrite', partitionBy='id')

    def apply_one_hot_encoding(self,
                               col: str):
        self.logger.info(f'One hot encoding of {col}')
        indexer = StringIndexer(inputCol=col, outputCol=f'{col}_num')
        df_indexed = indexer.fit(self.df).transform(self.df)
        encoder = OneHotEncoder(inputCol=f'{col}_num', outputCol=f'{col}_cat')
        df_one_hot = encoder.fit(df_indexed).transform(df_indexed)
        out_col = f'{col}_num'
        df_one_hot.drop(col)
        df_one_hot.drop(out_col)
        self.df = df_one_hot

    @staticmethod
    def get_raw_data_from_s3(s3_object: str, target_path: str):
        """
        Get s3_object from S3 and stored locally in target_path
        :param s3_object: Path to S3. e.g. s3://bucket-name/folder1/filename.ext.
        :param target_path: Folder name.
        """
        os.system(f'aws s3 cp {s3_object} {target_path}')

    def read_user_data(self, file_path: str, schema_obj):
        df = self.session.read.format('csv') \
            .option('delimiter', ',') \
            .option('quote', '"') \
            .option('header', 'true') \
            .option('encoding', 'UTF-16') \
            .schema(schema_obj) \
            .load(file_path)
        return df

    def replace_id_by_int(self,
                          col: str,
                          new_col: str):
        """
        Replaces heavy string column by a much lighter int column, when possible.
        The mapping is saved in data/processed/{new_col}
        :param col: Name of the column to be replaced.
        :param new_col: Name of the new (and much lighter) column.
        """
        self.logger.info(f'Replacing {col} by {new_col}')
        unique_ids = self.df.select(col).distinct().sort(col).withColumn(new_col, monotonically_increasing_id())
        unique_ids.write.parquet(path=f"../data/processed/{new_col}", mode='overwrite', partitionBy='id')
        df_join = self.df.join(unique_ids, on=col, how='left').drop(col)
        self.df = df_join

    @property
    def df(self):
        return self.__df

    @df.setter
    def df(self, new_df: pyspark.sql.DataFrame):
        self.__df = new_df

    @property
    def session(self):
        return self.__spark_session

    @property
    def logger(self):
        return self.__logger
