import pandas as pd
import numpy as np
import random
import cv2
from sklearn.model_selection import train_test_split

def load_data(img_size=90):

    train_dir = 'tmp-data'
    #train_dir = 'raw-data'
    train_df = pd.read_csv(train_dir+'/labels.csv')
    labels = np.asarray(pd.get_dummies(train_df['breed']))
    X_raw = []
    y_raw = []
    for i, fl in enumerate(train_df['id']):
        img = cv2.imread(train_dir+'/train/{}.jpg'.format(fl))
        X_raw.append(cv2.resize(img, dsize=(img_size, img_size)))
        y_raw.append(labels[i])

    # shapes of  X = (num_images, img_size, img_size, 3) and y = (num_images, num_class)
    X = np.array(X_raw, np.float32)/255
    y = np.array(y_raw, np.uint8)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    return X_train, X_test, y_train, y_test

