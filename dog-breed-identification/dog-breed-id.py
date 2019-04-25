import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils
import cv2
from utils import load_data

X_train, X_dev, y_train, y_dev = load_data()

#for image in X[0:4]: 
#   plt.imshow(image)
#   plt.show()

print(X_train.shape)
print(X_dev.shape)
print(y_train.shape)
print(y_dev.shape)

