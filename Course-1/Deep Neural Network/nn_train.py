# Package imports
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from nn_model import nn_model
from predict import predict
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X1 = cancer['data']
y1 = cancer['target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)

# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

X=X_train.T
Y=y_train.reshape(1,y_train.shape[0])
print(X.shape)
print(Y.shape)

#print("Start!")
# Build a model with a n_h-dimensional hidden layer
parameters = nn_model(X, Y, n_h = 10, num_iterations = 10000, print_cost=True)

# Print accuracy
predictions = predict(parameters, X)
print ('Training Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')

predictions = predict(parameters, X_test.T)
print ('Test Accuracy: %d' % float((np.dot(y_test,predictions.T) + np.dot(1-y_test,1-predictions.T))/float(y_test.size)*100) + '%')


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions.T))
print(classification_report(y_test,predictions.T))