from typing import List

import pandas as pd
import numpy as np

from enum import Enum
from collections import namedtuple
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold

from src.data.make_dataset import load_dataset
from src.features.build_features import get_dummy_values, fill_missing_values, FEATURE_COLUMNS

UID_VALC_COL = 'uid_val'

UID_TEST_COL = 'uid_test'

UID_TRAIN_COL = 'uid_train'

RATING = 'rating'

ITEM_ID_COL = 'item_id'
USER_ID_COL = 'user_id'
# COLUMNS_TO_DROP = [RATING, 'year', ITEM_ID_COL, 'split', 'category', 'timestamp', USER_ID_COL]
COLUMNS_TO_DROP = [RATING, ITEM_ID_COL, 'split', 'timestamp', USER_ID_COL]

ExperimentData = namedtuple(
    "ExperimentData",
    [
        "Xratings_train",
        "Xratings_test",
        "Xratings_valid",

        "Xfeatures_train",
        "Xfeatures_test",
        "Xfeatures_valid",

        "Xraw_train",
        "Xraw_test",
        "Xraw_valid",

        "test_uids",
        "test_uids_original",
        "test_iids",
        "test_y",
        "feature_names"
    ]
)


class DataType(Enum):
    TRAIN = 'train'
    TEST = 'test'
    VALID = 'valid'


def make_id_column_consecutive_integer_from_0(data, col, new_colname=None):
    le = LabelEncoder()
    data_encoded = data.copy()
    new_col = new_colname if new_colname else col
    data_encoded[new_col] = le.fit_transform(data_encoded[col])
    return data_encoded


def get_users_with_min_n_ratings(raw_data: pd.DataFrame, min_ratings_per_user: int = 3) -> pd.DataFrame:
    user_ids_ratings_cnt = pd.value_counts(raw_data.user_id).reset_index().rename(
        columns={'index': 'uid', 'user_id': 'cnt'})
    user_ids_min_ratings = user_ids_ratings_cnt.uid[user_ids_ratings_cnt.cnt >= min_ratings_per_user]
    unique_user_ids_min_ratings = user_ids_min_ratings.unique()

    data_min_n_ratings = raw_data.loc[raw_data.user_id.isin(unique_user_ids_min_ratings), :].copy()
    data_min_n_ratings = data_min_n_ratings.reset_index().drop("index", axis=1)

    data_min_n_ratings = make_id_column_consecutive_integer_from_0(data_min_n_ratings, 'user_id')
    data_min_n_ratings = make_id_column_consecutive_integer_from_0(data_min_n_ratings, 'item_id')

    assert set(data_min_n_ratings.user_id.unique()) == set(range(data_min_n_ratings.user_id.max() + 1))
    assert set(data_min_n_ratings.item_id.unique()) == set(range(data_min_n_ratings.item_id.max() + 1))

    return data_min_n_ratings


def prepare_sparse_ratings_matrix(data: pd.DataFrame, nuser: int, nitem: int, uid_colname: str,
                                  iid_colname: str = "item_id") -> csr_matrix:
    user_item_matrix = lil_matrix((nuser, nitem), dtype=np.float32)

    for row_idx, row in data.iterrows():
        uidx = row[uid_colname]
        iidx = row[iid_colname]
        rating = row[RATING]
        user_item_matrix[uidx, iidx] = rating
        if row_idx % 10000 == 0:
            print(f"Processed: {row_idx / float(data.shape[0])}%")

    user_item_matrix = user_item_matrix.tocsr()
    return user_item_matrix


def hide_test_data_ratings_for_prediction(Xr_test, X_test_raw, perc_test=0.3, min_ratings=3,
                                          uid_original_colname=USER_ID_COL, uid_consecutive_colname=UID_TEST_COL,
                                          iid_colname='item_id',
                                          random_state=999):
    Xr_test_pred_hidden = Xr_test.copy()
    X_test_raw_pred_hidden = X_test_raw.copy()
    row_nonzero, col_nonzero = Xr_test.nonzero()
    rating_counts = pd.value_counts(row_nonzero).reset_index().rename(columns={'index': 'uid', 0: 'cnt'})
    user_ids = rating_counts.uid[rating_counts.cnt > min_ratings]
    to_choose = int(np.round(len(user_ids) * perc_test, 0))
    np.random.seed(random_state)
    selected_uids = np.random.choice(user_ids, size=to_choose)

    uids_original = []
    uids_consecutive = []
    item_ids = []
    y = []

    for uid in selected_uids:
        user_data = X_test_raw.loc[X_test_raw[uid_consecutive_colname] == uid, :]
        item_id = int(np.random.choice(user_data[iid_colname], size=1)[0])
        uid_original = user_data[uid_original_colname].unique()[0]
        uids_consecutive.append(uid)
        uids_original.append(uid_original)
        item_ids.append(item_id)
        rating_sparse = Xr_test[uid, item_id].flatten()[0]
        rating_raw = user_data.loc[user_data.item_id == item_id, "rating"].iloc[0]

        assert rating_raw == rating_sparse

        y.append(rating_sparse)
        Xr_test_pred_hidden[uid, item_id] = 0.0
        X_test_raw_pred_hidden.loc[
            (X_test_raw_pred_hidden[uid_consecutive_colname] == uid) & (
                    X_test_raw_pred_hidden[iid_colname] == item_id)] = 0.0

    return uids_consecutive, uids_original, item_ids, y, Xr_test_pred_hidden, X_test_raw_pred_hidden


def crossval_generator(data: pd.DataFrame,
                       test_size: float = 0.4,
                       test_seed: int = 123,
                       valid_seed: int = 456,
                       test_ratigs_perc_to_hide: float = 0.3,
                       test_ratings_to_hide_seed: int = 999,
                       nsplits: int = 10) -> List[ExperimentData]:
    data = make_id_column_consecutive_integer_from_0(data, 'user_id')
    data = make_id_column_consecutive_integer_from_0(data, 'item_id')
    data.sort_values(by=['user_id', 'item_id'], ascending=True)
    nuser, nitem = data.user_id.max() + 1, data.item_id.max() + 1

    kf = KFold(n_splits=nsplits, random_state=valid_seed)
    cv = 0
    for train_ids, test_val_ids in kf.split(data.user_id.unique()):
        print(f"Processing cv {cv}")
        n_train = len(train_ids) // 2
        np.random.seed(123)
        np.random.shuffle(train_ids)
        train_ids = train_ids[:n_train]
        cv += 1
        yield prepare_experiment_data_from_sample(data, nitem, nuser, test_ratigs_perc_to_hide,
                                                  test_ratings_to_hide_seed,
                                                  test_seed, test_size, test_val_ids, train_ids)


def train_validation_test_split(
        data: pd.DataFrame,
        test_size: float = 0.4,
        valid_size: float = 0.5,
        test_seed: int = 123,
        valid_seed: int = 456,
        test_ratigs_perc_to_hide: float = 0.3,
        test_ratings_to_hide_seed: int = 999) -> ExperimentData:
    data = make_id_column_consecutive_integer_from_0(data, 'user_id')
    data = make_id_column_consecutive_integer_from_0(data, 'item_id')
    data.sort_values(by=['user_id', 'item_id'], ascending=True)
    nuser, nitem = data.user_id.max() + 1, data.item_id.max() + 1

    train_ids, test_val_ids = train_test_split(data.user_id.unique(), test_size=valid_size, random_state=valid_seed)

    return prepare_experiment_data_from_sample(data, nitem, nuser, test_ratigs_perc_to_hide, test_ratings_to_hide_seed,
                                               test_seed, test_size, test_val_ids, train_ids)


def prepare_experiment_data_from_sample(
        data,
        nitem,
        nuser,
        test_ratigs_perc_to_hide,
        test_ratings_to_hide_seed,
        test_seed,
        test_size,
        test_val_ids,
        train_ids,
        prepare_sparse_matrix=False):
    X_train_raw = data.loc[data.user_id.isin(train_ids)]
    test_ids, validation_ids = train_test_split(test_val_ids, test_size=test_size, random_state=test_seed)

    X_test_raw, X_val_raw = data.loc[data.user_id.isin(test_ids)], \
                            data.loc[data.user_id.isin(validation_ids)]

    raw_data_sets = {
        DataType.TRAIN: (X_train_raw, UID_TRAIN_COL),
        DataType.TEST: (X_test_raw, UID_TEST_COL),
        DataType.VALID: (X_val_raw, UID_VALC_COL)
    }
    processed_data = {}
    ratings_data = {}
    features_data = {}

    feature_names = []

    for dtype, (dset, new_uid_col) in raw_data_sets.items():
        processed_raw = make_id_column_consecutive_integer_from_0(dset.reset_index(drop=True), USER_ID_COL, new_uid_col)
        ratings_ds = prepare_sparse_ratings_matrix(processed_raw, nuser, nitem, new_uid_col)  #if prepare_sparse_ratings_matrix else None
        features_ds = processed_raw.drop(COLUMNS_TO_DROP + [new_uid_col], axis=1)
        if feature_names == []:
            feature_names = list(features_ds.columns)
        features_matrix = features_ds.to_numpy()

        processed_data[dtype] = processed_raw
        ratings_data[dtype] = ratings_ds
        features_data[dtype] = features_matrix

    assert sum([ds.shape[0] for ds in processed_data.values()]) == len(train_ids) + len(test_val_ids) #data.shape[0]

    uids, uids_original, item_ids, y, Xr_test_pred_hidden, X_test_raw_pred_hidden = hide_test_data_ratings_for_prediction(
        ratings_data[DataType.TEST],
        processed_data[DataType.TEST][[USER_ID_COL, UID_TEST_COL, ITEM_ID_COL, RATING]],
        perc_test=test_ratigs_perc_to_hide,
        random_state=test_ratings_to_hide_seed,
        uid_consecutive_colname=UID_TEST_COL
    )
    return ExperimentData(
        ratings_data[DataType.TRAIN],
        Xr_test_pred_hidden,
        ratings_data[DataType.VALID],

        features_data[DataType.TRAIN],
        features_data[DataType.TEST],
        features_data[DataType.VALID],

        processed_data[DataType.TRAIN][[USER_ID_COL, UID_TRAIN_COL, ITEM_ID_COL, RATING]],
        X_test_raw_pred_hidden,
        processed_data[DataType.VALID][[USER_ID_COL, UID_VALC_COL, ITEM_ID_COL, RATING]],

        uids,
        uids_original,
        item_ids,
        y,
        feature_names
    )


def prepare_experiment(download_new=False, test_seed: int = 123, valid_seed: int = 456) -> ExperimentData:
    raw_data = load_dataset(download_new)
    raw_data = get_dummy_values(fill_missing_values(raw_data))
    data_min_n_ratings = get_users_with_min_n_ratings(raw_data, 3)
    experiment_data = train_validation_test_split(data_min_n_ratings, test_seed=test_seed, valid_seed=valid_seed)
    return experiment_data


def prepare_crossval_experiment(download_new=False, test_size: float = 0.4, test_seed: int = 123, valid_seed: int = 456, nsplits: int = 10):
    raw_data = load_dataset(download_new)
    raw_data = get_dummy_values(fill_missing_values(raw_data))
    data_min_n_ratings = get_users_with_min_n_ratings(raw_data, 3)
    for exd in crossval_generator(raw_data, test_size, test_seed, valid_seed, nsplits=nsplits):
        yield exd
