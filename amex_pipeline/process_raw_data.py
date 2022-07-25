from pyspark.sql import SparkSession


def read_user_data(file_path, schema_obj):
    spark = SparkSession.builder.master("local[1]").appName("kaggle").getOrCreate()
    df = spark.read.format('csv')\
        .option('delimiter', ',')\
        .option('quote', '"')\
        .option('header', 'true') \
        .option('encoding', 'UTF-16') \
        .schema(schema_obj)\
        .load(file_path)
    return df
