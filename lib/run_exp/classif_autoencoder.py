import os
import numpy as np
import pandas as pd
import tensorflow_addons as tfa
from tensorflow import keras

# Define custom loss
def reconstruction_loss(encoder_input, decoder_output):

    # Create a loss function that is the MSE loss between augmented and decoded layers
    def loss(y_true, y_pred):
        return keras.losses.MeanSquaredError()(encoder_input.output, decoder_output.output)
        # return 0
   
    # Return a function
    return loss


def run_experiment(model, encoder_input, decoder_output,
                  ds_train, ds_test, ds_valid,
                  batch_size=32, num_epochs=100,
                  learning_rate=1e-3, weight_decay=1e-4,
                  lam_recon=10,
                  output_path=None, prefix='cnn',
                  from_logits=False, label_smoothing=0.1):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    model.compile(
        optimizer=optimizer,
        loss=[
            keras.losses.CategoricalCrossentropy(from_logits=from_logits, label_smoothing=label_smoothing),
            reconstruction_loss(encoder_input, decoder_output),
        ],
        loss_weights=[1., lam_recon],
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
        ],
    )

    # callbacks
    log_filename = os.path.join(output_path, prefix + '_log.csv')
    ckpt_path = os.path.join(output_path, 'ckpt')
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path, exist_ok=True)
    # checkpoint_filename = os.path.join(ckpt_path, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5')
    checkpoint_filename = os.path.join(ckpt_path, prefix + '_weights.hdf5')

    log = keras.callbacks.CSVLogger(log_filename)
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filename,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )
    custom_early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        min_delta=0.005,
        mode='max'
    )


    card = ds_train.cardinality().numpy()
    ds_shuffle = ds_train.shuffle(card, reshuffle_each_iteration=True)
    history = model.fit(
        x=ds_shuffle.batch(batch_size),
        batch_size=batch_size,
        epochs=num_epochs,
        validation_data=ds_test.batch(batch_size),
        callbacks=[log, checkpoint_callback, custom_early_stopping],
    )

    model.load_weights(checkpoint_filename)
    _, accuracy = model.evaluate(ds_valid.batch(batch_size))
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    y_pred = model.predict(ds_valid.batch(batch_size), batch_size=batch_size)
    y_valid = ds_valid.map(lambda img, label: label)
    y_valid = np.stack(list(y_valid))

    y_pred_pd = pd.DataFrame(y_pred, columns=[0, 1, 2]).idxmax(1)
    y_valid_pd = pd.DataFrame(y_valid, columns=[0, 1, 2]).idxmax(1)

    conf_mat = pd.crosstab(y_valid_pd, y_pred_pd,
                            colnames=['Predicted'],
                            rownames=['Real'],
                            )

    return history, conf_mat