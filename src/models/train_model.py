import tensorflow as tf
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from sklearn.preprocessing import LabelEncoder

from src.models.training_utils import mse_masked, mape_masked, mae_masked


def get_array(series):
    return np.array([[element] for element in series])

def get_collabfiltering_model1(max_user, max_item, dim_embedddings=30):
    bias = 3
    # inputs
    w_inputs = tf.keras.layers.Input(shape=(1,), dtype='int32')
    w = tf.keras.layers.Embedding(max_item+1, dim_embedddings, name="items")(w_inputs)

    # context
    u_inputs = tf.keras.layers.Input(shape=(1,), dtype='int32')
    u = tf.keras.layers.Embedding(max_user+1, dim_embedddings, name="user")(u_inputs)
    o = tf.keras.layers.multiply([w, u])
    o = tf.keras.layers.Dropout(0.5)(o)
    o = tf.keras.layers.Flatten()(o)
    o = tf.keras.layers.Dense(1)(o)

    rec_model = tf.keras.models.Model(inputs=[w_inputs, u_inputs], outputs=o)
    #rec_model.summary()
    rec_model.compile(loss=[mse_masked], metrics=[mse_masked, mape_masked, mae_masked], optimizer='adam')

    return rec_model

def get_colabfiltering_model2( X_colab_all: pd.DataFrame, nusers: int, nitems: int, nfactors):
    K = nfactors
    initializer = tf.initializers.GlorotUniform()
    P = tf.Variable(initializer((nusers, K)))
    Q = tf.Variable(initializer((nitems, K)))

    # build model

    ratings = X_colab_all.astype(np.float64)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    losses = []

    for i in tqdm(range(20)):
        with tf.GradientTape() as t:
            pMultq = tf.matmul(P, Q, transpose_b=True)
            squared_deltas = tf.square(pMultq - ratings)
            loss = tf.reduce_sum(squared_deltas)
        grad = t.gradient(loss, [P, Q])
        optimizer.apply_gradients(zip(grad, [P, Q]))
        losses.append(loss)
        print(loss)


def build_shallow_autorec_single_input(X_shape):
    inp = tf.keras.layers.Input(shape=(X_shape,))
    drop1 = tf.keras.layers.Dropout(rate=0.2)(inp)
    enc = tf.keras.layers.Dense(X_shape // 8, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(
        drop1)
    drop2 = tf.keras.layers.Dropout(rate=0.2)(enc)
    out = tf.keras.layers.Dense(X_shape, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(drop2)
    model = tf.keras.models.Model(inputs=inp, outputs=out)
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss=[mse_masked], metrics=[mse_masked, mape_masked, mae_masked])

    print(model.summary())
    return model


def build_deep_autorec_single_input(X_shape):
    inp = tf.keras.layers.Input(shape=(X_shape,))
    drop1 = tf.keras.layers.Dropout(rate=0.2)(inp)
    enc1 = tf.keras.layers.Dense(X_shape // 4, activation='relu')(drop1)
    drop2 = tf.keras.layers.Dropout(rate=0.2)(enc1)
    enc2 = tf.keras.layers.Dense(X_shape // 8, activation='relu')(drop2)
    drop3 = tf.keras.layers.Dropout(rate=0.2)(enc2)
    dec1 = tf.keras.layers.Dense(X_shape // 4, activation='relu')(drop3)
    drop4 = tf.keras.layers.Dropout(rate=0.2)(dec1)
    out = tf.keras.layers.Dense(X_shape, activation='relu')(drop4)
    model = tf.keras.models.Model(inputs=inp, outputs=out)
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss=[mse_masked], metrics=[mse_masked, mape_masked, mae_masked])

    print(model.summary())
    return model

def build_autorec_multi_input(X_shape, F_shape):
    inp1 = tf.keras.layers.Input(shape=(X_shape,))
    inp2 = tf.keras.layers.Input(shape=(F_shape,))
    concat = tf.keras.layers.Concatenate()
    combined = concat([inp1, inp2])
    drop1 = tf.keras.layers.Dropout(rate=0.1)(combined)
    # bnorm = tf.keras.layers.BatchNormalization()(drop1)
    enc = tf.keras.layers.Dense(
        X_shape // 4,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.001))(drop1)
    drop2 = tf.keras.layers.Dropout(rate=0.1)(enc)
    out = tf.keras.layers.Dense(
        X_shape,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.001))(drop2)
    model = tf.keras.models.Model(inputs=[inp1, inp2], outputs=out)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss=[mse_masked], metrics=[mse_masked, mape_masked, mae_masked])

    print(model.summary())
    return model

def build_autorec_multi_input2(X_shape, F_shape):
    inp1 = tf.keras.layers.Input(shape=(X_shape,))
    inp2 = tf.keras.layers.Input(shape=(F_shape,))
    concat = tf.keras.layers.Concatenate()
    drop1 = tf.keras.layers.Dropout(rate=0.1)(inp1)
    # bnorm = tf.keras.layers.BatchNormalization()(drop1)
    enc = tf.keras.layers.Dense(
        X_shape // 4,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.001))(drop1)
    combined = concat([enc, inp2])
    drop2 = tf.keras.layers.Dropout(rate=0.3)(combined)
    enc2 = tf.keras.layers.Dense(
        X_shape // 16,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.001))(drop2)
    drop3 = tf.keras.layers.Dropout(rate=0.3)(enc2)
    dec = tf.keras.layers.Dense(
        X_shape // 4,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.001))(drop3)
    drop4 = tf.keras.layers.Dropout(rate=0.3)(dec)
    out = tf.keras.layers.Dense(
        X_shape,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.001))(drop4)
    model = tf.keras.models.Model(inputs=[inp1, inp2], outputs=out)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
    model.compile(optimizer=optimizer, loss=[mse_masked], metrics=[mse_masked, mape_masked, mae_masked])

    print(model.summary())
    return model