import tensorflow as tf
from sklearn.utils import shuffle

def generator_ratings_features(ratings, features, normalize=False, batch_size=64):
    mask = (ratings > 0.0) * 1.0
    while True:
        ratings, mask = shuffle(ratings, mask)
        for i in range(ratings.shape[0] // batch_size + 1):
            upper = min((i + 1) * batch_size, ratings.shape[0])
            r = ratings[i * batch_size:upper].toarray()
            f = features[i * batch_size: upper]
            m = mask[i * batch_size:upper].toarray()
            if normalize:
                # r = r - mu * m
                r = r * m
            yield [r, f], r

def generator_ratings(ratings, normalize=False, batch_size=64):
    mask = (ratings > 0.0) * 1.0
    while True:
        ratings, mask = shuffle(ratings, mask)
        print("shuffling the data")
        for i in range(ratings.shape[0] // batch_size + 1):
            upper = min((i + 1) * batch_size, ratings.shape[0])
            r = ratings[i * batch_size:upper].toarray()
            m = mask[i * batch_size:upper].toarray()
            if normalize:
                r = r * m
            yield r, r

def mse_masked(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, 0.0), dtype='float32')
    y_true = y_true * mask
    y_pred = y_pred * mask
    diff = y_pred - y_true
    sqdiff = diff * diff * mask
    sse = tf.reduce_sum(tf.reduce_sum(sqdiff))
    n = tf.reduce_sum(tf.reduce_sum(mask))
    return sse / n

def mape_masked(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, 0), dtype='float32')

    y_true = y_true * mask
    y_pred = y_pred * mask

    return tf.reduce_mean(tf.abs((y_true[y_true > 0.0] - y_pred[y_true > 0.0]) / y_true[y_true > 0.0]))

def mae_masked(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, 0), dtype='float32')
    mape = tf.keras.losses.MeanAbsoluteError()
    return mape(y_true * mask, y_pred * mask)