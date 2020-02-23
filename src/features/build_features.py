import pandas as pd

FEATURE_COLUMNS = ['user_attr', 'model_attr', 'brand']

def fill_missing_values(raw_data: pd.DataFrame) -> pd.DataFrame:
    cols_to_fill = ['model_attr', 'user_attr', 'brand']
    for col in cols_to_fill:
        raw_data[col].fillna("missing", inplace=True)
    return raw_data

def get_dummy_values(raw_data: pd.DataFrame) -> pd.DataFrame:
    other_cols = set(raw_data.columns).difference(FEATURE_COLUMNS)
    encoded_data = pd.get_dummies(raw_data[FEATURE_COLUMNS], sparse=True)
    for other_col in other_cols:
        encoded_data[other_col] = raw_data[other_col]
    return encoded_data