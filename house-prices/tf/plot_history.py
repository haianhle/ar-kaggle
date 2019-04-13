import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_history(hist):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Squared Error')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,0.1])
  plt.legend()
  plt.show()

history = pd.read_csv('history.csv')
plot_history(history)

