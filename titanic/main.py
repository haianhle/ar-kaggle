'''
Binary classificiation project
Titanic passengers' survival using logistic regression

Author: Anh Reynolds
'''

# import libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import keras as k

from sklearn import preprocessing

# import data from csv files  
train_set = pd.read_csv('data/train.csv', delimiter=',', quotechar='"')   
test_set = pd.read_csv('data/test.csv', delimiter=',', quotechar='"')   
  
# Find important values  
# n_train = number of training examples  
# n_test  = number of test examples  
# n_x = number of input features (n_x is 1 smaller for test_set because we don't know the survival)  
  
n_train = train_set.shape[0]  
n_test = test_set.shape[0]  
n_x = train_set.shape[1]-1  
  
print("Number of training examples: n_train = " + str(n_train))  
print("Number of testing examples: n_test = " + str(n_test))  
print("Number of input features for each example: n_x = " + str(n_x))  

# missing data
print('Missing data in each feature in training set:\n', train_set.isnull().sum())
print('Missing data in each feature in test set:\n', test_set.isnull().sum())

# replacing missing values in 'Age' with median age
median_age = train_set['Age'].median()
train_set['Age'].fillna(median_age, inplace=True)

median_age = test_set['Age'].median()
test_set['Age'].fillna(median_age, inplace=True)

# extract processed data
# Extract data from train_set
train_features = [train_set['Pclass'], train_set['Sex'], train_set['Age'], train_set['SibSp'], train_set['Parch'], train_set['Fare']]
X1 = pd.concat(train_features, axis=1)
n_features = X1.shape[1]
print('Number of features to train = ', n_features)

#Extract data into X_train and Y_train, now numpy arrays -- not a good way!!!!
X_train = X1.iloc[:, :].values #matrix dimensions (n_train, n_x-4) excluded 'Name', 'Ticket', 'Cabin', 'Embarked'
Y_train = train_set.iloc[:, 1].values #vector dimension (n_train)
gender = preprocessing.LabelEncoder()
X_train[:, 1] = gender.fit_transform(X_train[:, 1])

#Extract data from train_set
test_features = [test_set['Pclass'], test_set['Sex'], test_set['Age'], test_set['SibSp'], test_set['Parch'], test_set['Fare']]
X2 = pd.concat(test_features, axis=1)

X_test = X2.iloc[:, :].values
X_test[:, 1] = gender.fit_transform(X_test[:, 1])
  
# normalizing inputs - appears to me so far using mean and variance 
sc = preprocessing.StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
# need X to have shape (n_features, n_examples)
X_trainT = np.transpose(X_train)
X_testT = np.transpose(X_test)

# attempt to use Keras to train a fully connected network
model = k.models.Sequential()

#add input layer and the first hidden layer
model.add(k.layers.Dense(5, kernel_initializer='uniform', input_shape=(n_features,), activation='relu'))

#add second hidden layer
model.add(k.layers.Dense(5, kernel_initializer='uniform', activation='relu'))

#add output layer
model.add(k.layers.Dense(1, kernel_initializer='uniform', activation='sigmoid'))

#compile network
optimizer = k.optimizers.Adam(lr=0.005)
model.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])

#run network
model.fit(X_train, Y_train, batch_size=100, epochs=10)

Y_pred = model.predict(X_test, batch_size=100)
Y_pred = (Y_pred > 0.5).astype(int)
Y = pd.DataFrame({'Survived' : Y_pred[:, 0]})
submission = pd.concat((test_set['PassengerId'], Y), axis=1)
submission.head(100)
print('Number of passengers predicted to survive =', Y.sum())
submission.to_csv('data/submission_AR.csv', index=False)
