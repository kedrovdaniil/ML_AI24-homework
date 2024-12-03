import re

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

df_train = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')
df_train.drop_duplicates(subset=df_train.columns.difference(['selling_price']), keep='first', inplace=True)
X_train = df_train.drop(columns=['selling_price'])
y_train = df_train['selling_price']


def parse_num_value(value):
    if not isinstance(value, str):
        return None

    num = re.search(r'(\d+(\.\d+)?)', value.lower().strip())

    if num:
        return num.group(1)

    return None


class DropFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_)
        return X.drop(columns=self.columns_to_drop, errors='ignore')


class ParseNumValues(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_)
        X['mileage'] = X['mileage'].apply(parse_num_value)
        X['engine'] = X['engine'].apply(parse_num_value)
        X['max_power'] = X['max_power'].apply(parse_num_value)
        return X


class Fill_NA(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_)

        median_columns = ['mileage', 'engine', 'max_power', 'seats']
        X_cleaned = X.dropna(subset=median_columns)
        X_cleaned['mileage'] = X_cleaned['mileage'].astype(float)
        X_cleaned['engine'] = X_cleaned['engine'].astype(float)
        X_cleaned['max_power'] = X_cleaned['max_power'].astype(float)

        for column in median_columns:
            median = X_cleaned[column].median()
            X[column].fillna(median, inplace=True)
            X[column].fillna(median, inplace=True)

        return X


class RemoveDuplicatesAndChangeTypes(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_)

        X.drop_duplicates(subset=X.columns.difference(['selling_price']), keep='first')
        # X = X.iloc[valid_indices]

        X.reset_index(drop=True, inplace=True)

        X['mileage'] = X['mileage'].astype(float)
        X['engine'] = X['engine'].astype(float)
        X['max_power'] = X['max_power'].astype(float)
        X['engine'] = X['engine'].astype(int)
        return X


categorical_features = ['fuel', 'seller_type', 'transmission', 'owner']
numeric_features = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features),
    ]
)

remove_duplicates = RemoveDuplicatesAndChangeTypes()

# Создание пайплайна
model = Pipeline(steps=[
    ('drop_orig_name_feature', DropFeatureTransformer(columns_to_drop=['name', 'torque'])),
    ('parse_num_values', ParseNumValues()),
    ('fill_na_with_median', Fill_NA()),
    ('remove_duplicates_and_change_types', remove_duplicates),
    ('preprocessor', preprocessor),
    ('drop_orig_categorical_features', DropFeatureTransformer(columns_to_drop=categorical_features)),
    ('regressor', ElasticNet(alpha=1, l1_ratio=0.9)),
])

model.fit(X_train, y_train)

# Сохранение пайплайна
joblib.dump(model, 'elasticnet_pipeline.joblib')
