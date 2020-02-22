import io
import os

import pandas as pd
import requests

DATA_PATH = os.path.join(os.curdir, "data", "processed")
RAW_DATA_FILENAME = os.path.join(DATA_PATH, "raw_data.csv")

def download_dataset() -> pd.DataFrame:
    url = "https://raw.githubusercontent.com/MengtingWan/marketBias/master/data/df_electronics.csv"
    s = requests.get(url).content
    raw_data = pd.read_csv(io.StringIO(s.decode('utf-8')), sep=",")
    raw_data.to_csv(RAW_DATA_FILENAME, index=False)
    return raw_data

def load_dataset(download_new: bool=False) -> pd.DataFrame:
    raw_data = download_dataset() if download_new else pd.read_csv(RAW_DATA_FILENAME)
    return raw_data

def validate_unique_ids(raw_data: pd.DataFrame):
    uids_nums = set(range(raw_data.user_id.max() + 1))
    uids_set = set(raw_data.user_id.unique())
    assert uids_nums == uids_set

    item_ids_nums = set(range(raw_data.item_id.max() + 1))
    item_ids_set = set(raw_data.item_id.unique())
    assert item_ids_nums == item_ids_set
    max_uid = raw_data.user_id.max()

    unique_uid = pd.Series(raw_data.user_id.unique())
    expected_ids = pd.Series(range(max_uid + 1))

    pd.util.testing.assert_series_equal(unique_uid, expected_ids)
