from typing import Tuple
from abc import ABC, abstractmethod
from functools import reduce
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import AUC, mean_squared_error
from tensorflow.keras.layers import Input, Dense
from pyarrow.parquet import ParquetFile
import pyarrow as pa


class AmexModel(ABC):
    ID_NAME: str = 'customer_ID'
    CAT_VAR = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120',
               'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
    TARGET_NAME: str = 'prediction'
    DATE_NAME: str = 'S_2'
    LABEL_NAME: str = 'target'
    PATH_TO_TRAIN_DATA: str = './data/processed/archive/train.parquet'
    PATH_TO_TEST_DATA: str = './data/processed/archive/test.parquet'
    PATH_TO_LABELS: str = './data/train_labels.csv'
    NUM_FEATURES: int = 188
    BATCH_TEST_SIZE: int = 1000000
    BATCH_TEST_LIMIT: int = 11000001
    BATCH_TRAIN_SIZE: int = 14000
    EPOCHS: int = 20

    def __init__(self, model_name: str, **kwargs):
        self.agg_data = kwargs['agg_data']
        self.model_name = model_name
        self.__model = self.create_model(**kwargs)
        self.__train_data, self.__target = self.prepare_data()

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        object_ids = pd.read_csv(self.PATH_TO_LABELS)
        object_ids.set_index(self.ID_NAME, inplace=True)
        df = pd.read_parquet(self.PATH_TO_TRAIN_DATA)
        if self.agg_data:
            df = df.groupby(self.ID_NAME).mean()
        else:
            df.drop(self.DATE_NAME, axis=1, inplace=True)
            # df.drop(self.CAT_VAR, axis=1, inplace=True)
            df.set_index(self.ID_NAME, inplace=True)
        df = df.fillna(0)
        df = df.join(object_ids)
        target = df[[self.LABEL_NAME]].values
        df.drop(self.LABEL_NAME, axis=1, inplace=True)
        return df, target

    def generate_prediction(self):
        list_df = []
        for i in range(0, self.BATCH_TEST_LIMIT, self.BATCH_TEST_SIZE):
            list_df.append(pd.read_csv(f'prediction_{i}.csv'))
        df = pd.concat(list_df)
        df = df.groupby(self.ID_NAME).mean()
        df[self.TARGET_NAME] = (df.target > 0.5).astype('int')
        df.drop(self.LABEL_NAME, axis=1, inplace=True)
        [os.remove(f'prediction_{item}.csv') for item in range(0, self.BATCH_TEST_LIMIT, self.BATCH_TEST_SIZE)]
        df.to_csv(f'prediction_{self.model_name}.csv')

    def predict(self):
        try:
            trained_model = tf.keras.models.load_model(self.model_name)
        except OSError:
            trained_model = self.model
        pf = ParquetFile(self.PATH_TO_TEST_DATA)
        pf_it = pf.iter_batches(batch_size=self.BATCH_TEST_SIZE)
        for i in range(0, pf.metadata.num_rows, self.BATCH_TEST_SIZE):
            first_rows = next(pf_it)
            df_test = pa.Table.from_batches([first_rows]).to_pandas()
            if self.agg_data:
                df_test = df_test.groupby(self.ID_NAME).mean()
            else:
                df_test.drop(self.DATE_NAME, axis=1, inplace=True)
                df_test.set_index(self.ID_NAME)
            df_test = df_test.fillna(0)
            df = np.array(df_test.values)
            try:
                predictions = trained_model.predict(df)
            except TypeError:
                predictions = trained_model.predict(xgb.DMatrix(df_test))
            df_test[self.LABEL_NAME] = predictions
            df_test[[self.LABEL_NAME]].to_csv(f'prediction_{i}.csv')
        self.generate_prediction()

    @abstractmethod
    def create_model(self, **kwargs):
        pass

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, new_model):
        self.__model = new_model

    @property
    def train_data(self):
        return self.__train_data

    @property
    def target(self):
        return self.__target


class NaiveModel(AmexModel):
    def __init__(self, model_name: str):
        super().__init__(model_name=model_name, agg_data=True)

    def create_model(self, **kwargs):
        my_input = Input(shape=(self.NUM_FEATURES,))
        encoder = Dense(128, activation='relu')(my_input)
        encoder = Dense(64, activation='relu')(encoder)
        encoder = Dense(32, activation='relu')(encoder)
        encoder = Dense(16, activation='relu')(encoder)
        classifier = Dense(1, activation='sigmoid')(encoder)
        model_clf = Model(inputs=my_input, outputs=classifier)
        model_clf.compile(optimizer='adam',
                          loss='binary_crossentropy',
                          metrics=['accuracy', AUC()])
        return model_clf

    def train_model(self):
        df_values = np.array(self.train_data.values)
        self.model.fit(df_values, self.target,
                       batch_size=self.BATCH_TRAIN_SIZE,
                       epochs=self.EPOCHS,
                       verbose=2)
        self.model.save(self.model_name)


class AutoencoderModel(AmexModel):
    def __init__(self, model_name: str):
        super().__init__(model_name=model_name, agg_data=True)

    def train_model(self):
        df_values = np.array(self.train_data.values)
        self.model[0].fit(df_values, df_values,
                          batch_size=self.BATCH_TRAIN_SIZE,
                          epochs=self.EPOCHS, verbose=2)
        self.model[1].fit(df_values, self.target,
                          batch_size=self.BATCH_TRAIN_SIZE,
                          epochs=self.EPOCHS, verbose=2)
        self.model = self.model[1]
        self.model.save(self.model_name)

    def create_model(self, **kwargs):
        my_input = Input(shape=(self.NUM_FEATURES,))
        encoder = Dense(128, activation='relu')(my_input)
        encoder = Dense(64, activation='relu')(encoder)
        encoder = Dense(32, activation='relu')(encoder)
        encoder = Dense(16, activation='relu')(encoder)
        encoder = Dense(8, activation='relu')(encoder)
        decoder = Dense(16, activation='relu')(encoder)
        decoder = Dense(32, activation='relu')(decoder)
        decoder = Dense(64, activation='relu')(decoder)
        decoder = Dense(128, activation='relu')(decoder)
        decoder = Dense(self.NUM_FEATURES, activation='relu')(decoder)
        model_autoencoder = Model(inputs=my_input, outputs=decoder)
        model_autoencoder.compile(optimizer='adam',
                                  loss='mean_squared_error',
                                  metrics=['mean_squared_error'])

        classifier = Dense(1, activation='sigmoid')(encoder)
        # tie it together
        model_clf = Model(inputs=my_input, outputs=classifier)
        model_clf.compile(optimizer='adam',
                          loss='binary_crossentropy',
                          metrics=['accuracy', AUC()])
        return model_autoencoder, model_clf


class AggXGBoost(AmexModel):
    def __init__(self, model_name: str):
        super().__init__(model_name=model_name, agg_data=False)

    def train_model(self):
        if self.agg_data:
            self.train_model_stream()
        else:
            self.train_model_batch()

    def train_model_stream(self):
        train_data = self.train_data
        train_data[self.LABEL_NAME] = self.target
        train, test = train_test_split(self.train_data,
                                       test_size=.1, random_state=1999)
        train_mat = xgb.DMatrix(train.drop(self.LABEL_NAME, 1),
                                label=train[self.LABEL_NAME])
        test_mat = xgb.DMatrix(test.drop(self.LABEL_NAME, 1),
                               label=test[self.LABEL_NAME])
        evaluation = [(test_mat, "eval"), (train_mat, "train")]
        model = xgb.train(self.model, train_mat, 100, evaluation)
        self.model = model
        self.model.save_model(self.model_name)

    def train_model_batch(self):
        train_data = self.train_data
        train_data[self.LABEL_NAME] = self.target
        train, test = train_test_split(self.train_data,
                                       test_size=.1, random_state=1999)
        previous = 0
        batch = 100000
        test_mat = xgb.DMatrix(test.drop(self.LABEL_NAME, 1),
                               label=test[self.LABEL_NAME])
        self.model.update({'process_type': 'update',
                           'updater': 'refresh',
                           'refresh_leaf': True})
        for i in range(batch, train.size, batch):
            train_mat = xgb.DMatrix(train.iloc[previous:i, :].drop(self.LABEL_NAME, 1),
                                    label=train.iloc[previous:i, self.LABEL_NAME])
            previous = i
            evaluation = [(test_mat, "eval"), (train_mat, "train")]
            if i > 100000:
                model = xgb.train(self.model, train_mat, 100, evaluation, xgb_model=model)
            else:
                model = xgb.train(self.model, train_mat, 100, evaluation)
        self.model = model
        self.model.save_model(self.model_name)

    def create_model(self, **kwargs):
        parameters = {"booster": "gbtree", "max_depth": 2, "eta": 0.3,
                      "objective": "binary:logistic", "nthread": 2}
        return parameters


def create_ensemble_prediction(prediction_path: str):
    index_id = 'customer_ID'
    predictions = {0.498: 'prediction_median.csv',
                   0.510: 'prediction_naive.csv',
                   0.540: 'prediction_autoencoder_mea_01.csv',
                   0.541: 'prediction_autoencoder_mean.csv'}
    files = list()
    for weight, path_file in predictions.items():
        path_file = os.path.join(prediction_path, path_file)
        df = pd.read_csv(path_file)
        df.set_index(index_id, inplace=True)
        df = df*weight
        files.append(df)
    weight_sum = sum(predictions.keys())
    df = reduce(lambda a, b: a+b, files)/weight_sum
    df['prediction'] = (df.prediction > 0.5).astype('int')
    df.to_csv('model_ensemble.csv')
    return df


if __name__ == '__main__':
    model_name_ = 'xgboost_class'
    mh = AggXGBoost(model_name=model_name_)
    mh.train_model()
    mh.predict()
    pass
