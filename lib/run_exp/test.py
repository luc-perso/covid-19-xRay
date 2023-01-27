import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn import metrics


def test_accuracy(model, ds_test, batch_size):
  _, accuracy = model.evaluate(ds_test.batch(batch_size))
  print(f"Test accuracy: {round(accuracy * 100, 2)}%")

  return accuracy


def test_conf_mat(model, ds_test, batch_size):
  y_pred = model.predict(ds_test.batch(batch_size), batch_size=batch_size)
  y_test = ds_test.map(lambda img, label: label)
  y_test = np.stack(list(y_test))

  y_pred_pd = pd.DataFrame(y_pred, columns=[0, 1, 2]).idxmax(1)
  y_test_pd = pd.DataFrame(y_test, columns=[0, 1, 2]).idxmax(1)

  conf_mat = pd.crosstab(y_test_pd, y_pred_pd, 
                          colnames=['Predicted'],
                          rownames=['Real'],
                          )

  print(metrics.classification_report(y_test_pd, y_pred_pd))

  return conf_mat


def test_model(model, ds_test, batch_size):
  test_accuracy(model, ds_test, batch_size)

  conf_mat = test_conf_mat(model, ds_test, batch_size)
  
  return conf_mat


def compile_test_model(model, ds_test, batch_size, from_logits=False, label_smoothing=0.1):
  model.compile(
    loss=keras.losses.CategoricalCrossentropy(from_logits=from_logits, label_smoothing=label_smoothing),
    metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")]
  )
  conf_mat = test_model(model, ds_test, batch_size)

  return conf_mat