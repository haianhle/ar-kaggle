import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils
import cv2
from utils import load_data, load_test_data
import keras
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, Flatten

IMG_SIZE = 90
X_train, X_dev, y_train, y_dev = load_data(IMG_SIZE)
X_test = load_test_data(IMG_SIZE)
num_class = y_train.shape[1]

#for image in X[0:4]: 
#   plt.imshow(image)
#   plt.show()

print(X_train.shape)
print(X_dev.shape)
print(y_train.shape)
print(y_dev.shape)
print(X_test.shape)

base_model = VGG16(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
# Add a new top layer
x = base_model.output
x = Flatten()(x)
predictions = Dense(num_class, activation='softmax')(x)

# This is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# train only the top layers (which were randomly initialized)
for layer in base_model.layers:
    layer.trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]
model.summary()

model.fit(X_train, y_train, epochs=1, validation_data=(X_dev, y_dev), verbose=1)
preds = model.predict(X_test, verbose=1)

# Submission
sub = pd.DataFrame(preds)
# Set column names 
df_train = pd.read_csv('raw_data/labels.csv')
df_test = pd.read_csv('raw_data/sample_submission.csv')
one_hot = pd.get_dummies(pd.Series(df_train['breed']), sparse = True)
sub.columns =  one_hot.columns.values
# Insert the column id from the sample_submission at the start of the data frame
sub.insert(0, 'id', df_test['id'])
sub.head(5)
sub.to_csv('submission_AR.csv', index=False)
