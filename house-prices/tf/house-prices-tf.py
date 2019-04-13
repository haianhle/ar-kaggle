import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler

def loadData():
    X = pd.read_csv('data/X.csv', header=None)
    y = pd.read_csv('data/y.csv', header=None)
    y = np.log(y+1)
    X_test = pd.read_csv('data/X_test.csv', header=None)

    return X, y, X_test

def normData(x1, x2):
    return  x1_norm, x2_norm

def buildModel():
    model = tf.keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=(X.shape[1], )),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
    ])
    
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
    
    return model

def plot_history(hist):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Squared Error')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()
  plt.show()



###############

X, y, X_test = loadData()
X_norm, X_test_norm = normData(X, X_test) 

model = buildModel()

class PrintDot(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

history = model.fit(X_norm, y, epochs=1200, validation_split = 0.2, verbose=0, callbacks=[PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.to_csv('history.csv', index=False)

loss, mae, mse = model.evaluate(X_norm, y, verbose=0)
print("Training set Mean Abs Error: {:5.2f}".format(mae))
print("Training set Mean Sq Error: {:5.2f}".format(mse))

#plot_history(hist)

y_pred = model.predict(X_norm)
y_pred_df = pd.DataFrame(data=y_pred, columns=['y_pred'])
y_pred_df.insert(loc=0, column='y', value=y)
y_pred_df.to_csv('train_results.csv', index=False)

log_error = np.sqrt(np.mean(np.power(y_pred-y, 2)))
error = np.sqrt(np.mean(np.power(np.exp(y_pred)-np.exp(y), 2)))
print('Train set Mean Squared Log Error: ', log_error)
print('Train set Mean Sq Error in price predictions: ', error)

#### Make predictions ####
y_test = model.predict(X_test_norm)
y_test = np.exp(y_test) - 1
submission = pd.DataFrame(data=y_test, columns=['Saleprice'])
test_file = pd.read_csv('data/test.csv')
submission.insert(loc=0, column='Id', value=test_file['Id'])
submission.to_csv('predictions.csv', index=False)
