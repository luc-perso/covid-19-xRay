import os
import tensorflow_addons as tfa
from tensorflow import keras

def run_experiment(model,
                  ds_train, ds_valid, ds_test,
                  batch_size=32, num_epochs=100,
                  learning_rate=1e-3, weight_decay=1e-4,
                  patience=5, min_delta=0.005,
                  log_path=None, ckpt_path=None,
                  prefix='reconstruction'):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error"),
        metrics=[keras.metrics.MeanSquaredError(name="mse", dtype=None)],
    )

    # check dir
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path, exist_ok=True)
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    # callbacks
    log_filename = os.path.join(log_path, prefix + '_log.csv')
    # checkpoint_filename = os.path.join(ckpt_path, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5')
    checkpoint_filename = os.path.join(ckpt_path, prefix + '_weights.hdf5')

    log = keras.callbacks.CSVLogger(log_filename)
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filename,
        monitor="val_mse",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )
    custom_early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_mse',
        patience=patience,
        min_delta=min_delta,
        mode='min'
    )

    card = ds_train.cardinality().numpy()
    ds_shuffle = ds_train.shuffle(card, reshuffle_each_iteration=True)
    history = model.fit(
        x=ds_shuffle.batch(batch_size),
        batch_size=batch_size,
        epochs=num_epochs,
        validation_data=ds_valid.batch(batch_size),
        callbacks=[log, checkpoint_callback, custom_early_stopping],
    )

    return history
