import pandas as pd
import numpy as np

from collections import namedtuple
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

ITEM_ID_COL = 'item_id'

USER_ID_COL = 'user_id'

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
        "test_iids",
        "test_y"
    ]
)

def get_users_with_min_n_ratings(raw_data: pd.DataFrame, min_ratings_per_user: int = 3) -> pd.DataFrame:
    user_ids_ratings_cnt = pd.value_counts(raw_data.user_id).reset_index().rename(
        columns={'index': 'uid', 'user_id': 'cnt'})
    user_ids_min_ratings = user_ids_ratings_cnt.uid[user_ids_ratings_cnt.cnt >= min_ratings_per_user]
    unique_user_ids_min_ratings = user_ids_min_ratings.unique()

    raw_data_min_3_ratings = raw_data.loc[raw_data.user_id.isin(unique_user_ids_min_ratings), :].copy()
    raw_data_min_3_ratings = raw_data_min_3_ratings.reset_index().drop("index", axis=1)

    uid_le = LabelEncoder()
    uid_encoded_consecutive = uid_le.fit_transform(raw_data_min_3_ratings.user_id)

    iid_le = LabelEncoder()
    iid_encoded_consecutive = iid_le.fit_transform(raw_data_min_3_ratings.item_id)

    raw_data_min_3_ratings.user_id = uid_encoded_consecutive
    raw_data_min_3_ratings.item_id = iid_encoded_consecutive

    assert set(raw_data_min_3_ratings.user_id.unique()) == set(range(raw_data_min_3_ratings.user_id.max() + 1))
    assert set(raw_data_min_3_ratings.item_id.unique()) == set(range(raw_data_min_3_ratings.item_id.max() + 1))

    return raw_data_min_3_ratings

def prepare_sparse_ratings_matrix(data: pd.DataFrame) -> csr_matrix:
    nuser, nitem = data.user_id.max() + 1, data.item_id.max() + 1
    user_item_matrix = lil_matrix((nuser, nitem), dtype=np.int8)

    for row_idx, row in data.iterrows():
        uidx = row['user_id']
        iidx = row['item_id']
        rating = row['rating']
        user_item_matrix[uidx, iidx] = rating
        if row_idx % 10000 == 0:
            print(f"Processed: {row_idx / float(data.shape[0])}%")

    user_item_matrix = user_item_matrix.tocsr()
    return user_item_matrix

def hide_test_data_ratings_for_prediction(Xr_test, X_test_raw, perc_test=0.3, min_ratings=3,
                                          random_state=999):
    Xr_test_pred_hidden = Xr_test.copy()
    X_test_raw_pred_hidden = X_test_raw.copy()
    row_nonzero, col_nonzero = Xr_test.nonzero()
    nonzero_coords = dict(zip(row_nonzero, col_nonzero))
    rating_counts = pd.value_counts(row_nonzero).reset_index().rename(columns={'index': 'uid', 0: 'cnt'})
    user_ids = rating_counts.uid[rating_counts.cnt > min_ratings]
    to_choose = int(np.round(len(user_ids) * perc_test, 0))
    np.random.seed(random_state)
    selected_uids = np.random.choice(user_ids, size=to_choose)

    uids = []
    item_ids = []
    y = []
    for uid in selected_uids:
        _, user_item_ids = Xr_test[uid, :].nonzero()
        item_id = np.random.choice(user_item_ids, size=1)
        uids.append(uid)
        item_ids.append(item_id[0])
        rating = Xr_test[uid, item_id].toarray().flatten()[0]
        rating_raw = X_test_raw.query(f'(user_id == {uid}) and (item_id == {item_id[0]})').iloc[0].rating

        assert rating_raw == rating

        y.append(rating)
        Xr_test_pred_hidden[uid, item_id] = 0
        X_test_raw_pred_hidden.loc[
            (X_test_raw_pred_hidden.user_id == uid) & (X_test_raw_pred_hidden.item_id == item_id[0])] = 0

    return uids, item_ids, y, Xr_test_pred_hidden, X_test_raw_pred_hidden

def train_validation_test_split(
        data: pd.DataFrame,
        ratings_matrix: csr_matrix,
        test_size: float = 0.4,
        valid_size: float = 0.5,
        test_seed: int = 123,
        valid_seed: int = 456,
        test_ratigs_perc_to_hide: float = 0.3,
        test_ratings_to_hide_seed: int = 999) -> ExperimentData:
    train_ids, test_val_ids = train_test_split(data.user_id.unique(), test_size=valid_size, random_state=valid_seed)

    X_train_raw = data.loc[data.user_id.isin(train_ids)]
    features_matrix = X_train_raw.drop([USER_ID_COL, ITEM_ID_COL], axis=1).to_numpy()

    test_ids, validation_ids = train_test_split(test_val_ids, test_size=test_size, random_state=test_seed)
    X_test_raw, X_val_raw = data.loc[data.user_id.isin(test_ids)], \
                            data.loc[data.user_id.isin(validation_ids)]

    assert set(X_test_raw.user_id.values).union(X_val_raw.user_id.values).union(X_train_raw.user_id) == set(
        data.user_id)
    assert X_train_raw.shape[0] + X_test_raw.shape[0] + X_val_raw.shape[0] == data.shape[0]

    Xr_train = ratings_matrix[train_ids, :]
    Xf_train = features_matrix[train_ids, :]

    Xr_test = ratings_matrix[test_ids, :]
    Xf_test = features_matrix[test_ids, :]

    Xr_val = ratings_matrix[validation_ids, :]
    Xf_val = features_matrix[validation_ids, :]

    uids, item_ids, y, Xr_test_pred_hidden, X_test_raw_pred_hidden = hide_test_data_ratings_for_prediction(
        Xr_test,
        X_test_raw,
        perc_test=test_ratigs_perc_to_hide,
        random_state=test_ratings_to_hide_seed
    )

    return ExperimentData(
        Xr_train,
        Xr_test_pred_hidden,
        Xr_val,

        Xf_train,
        Xf_test,
        Xf_val,

        X_train_raw,
        X_test_raw_pred_hidden,
        X_val_raw,

        uids,
        item_ids,
        y
    )
