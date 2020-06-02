import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Tuple

FEATURE_COLUMNS = ['user_attr', 'model_attr', 'brand', "category", "year"]

def fill_missing_values(raw_data: pd.DataFrame) -> pd.DataFrame:
    cols_to_fill = ['model_attr', 'user_attr', 'brand']
    for col in cols_to_fill:
        raw_data[col].fillna("missing", inplace=True)
    return raw_data

def get_dummy_values(raw_data: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    other_cols = set(raw_data.columns).difference(FEATURE_COLUMNS)
    print(other_cols)
    raw_data['year'] = raw_data['year'].astype('str')
    encoded_data = raw_data[FEATURE_COLUMNS].copy()
    encoders = {}

    for f in FEATURE_COLUMNS:
        le = LabelEncoder()
        encoded_data[f] = le.fit_transform(encoded_data[f])
        encoders[f] = le
    for other_col in other_cols:
        encoded_data[other_col] = raw_data[other_col]
    return encoded_data, encoders