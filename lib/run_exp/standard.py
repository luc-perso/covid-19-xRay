import tensorflow_addons as tfa
from tensorflow import keras


def run_experiment(model,
                  ds_train, ds_test, ds_valid,
                  batch_size=32, num_epochs=100,
                  learning_rate=1e-3, weight_decay=1e-4,
                  checkpoint_filepath=None,
                  from_logits=False, label_smoothing=0.1):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(from_logits=from_logits,
                                                label_smoothing=label_smoothing),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
        ],
    )

    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    card = ds_train.cardinality().numpy()
    ds_shuffle = ds_train.shuffle(card, reshuffle_each_iteration=True)
    history = model.fit(
        x=ds_shuffle.batch(batch_size),
        batch_size=batch_size,
        epochs=num_epochs,
        validation_data=ds_test.batch(batch_size),
        callbacks=[checkpoint_callback],
    )

    model.load_weights(checkpoint_filepath)
    _, accuracy = model.evaluate(ds_valid.batch(batch_size))
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history