import pandas as pd


column_mapping = {
    'customer_ID': 'StringType()',
    'S_2': 'DateType()',
    'B_30': 'StringType()',
    'B_38': 'StringType()',
    'D_114': 'StringType()',
    'D_116': 'StringType()',
    'D_117': 'StringType()',
    'D_120': 'StringType()',
    'D_126': 'StringType()',
    'D_63': 'StringType()',
    'D_64': 'StringType()',
    'D_66': 'StringType()',
    'D_68': 'StringType()'
}


def get_data_type(col: str):
    try:
        return column_mapping[col]
    except KeyError:
        return 'FloatType()'


def create_schema(df: pd.DataFrame):
    with open('schemae/schema_raw.py', 'w', encoding='utf-8') as fp:
        fp.write('from pyspark.sql.types import StructType, StructField, BinaryType, '
                 'StringType, IntegerType, FloatType, DateType\n\n\n')
        fp.write('schema_amex = StructType(\n\t[')
        for col in df.columns:
            print(f'parsing column: {col}')
            data_type = get_data_type(col)
            line_content = f'\n\t\tStructField("{col}", {data_type}, True),'
            fp.write(line_content)
        fp.write('\b\n\t]\n)\n')

