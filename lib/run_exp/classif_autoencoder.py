import os
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
                  ds_train, ds_valid, ds_test,
                  batch_size=32, num_epochs=100,
                  learning_rate=1e-3, weight_decay=1e-4,
                  lam_recon=10,
                  from_logits=False, label_smoothing=0.1,
                  patience=5, min_delta=0.005,
                  log_path=None, ckpt_path=None,
                  prefix='cnn'):
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
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )
    custom_early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=patience,
        min_delta=min_delta,
        mode='max'
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