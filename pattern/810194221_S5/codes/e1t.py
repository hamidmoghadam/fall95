import numpy as np
import scipy.io
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from scipy.stats import multivariate_normal


def retrive_label(data):
    return np.array([x[51:].tolist().index(1)+1 for x in data])


ds = scipy.io.loadmat('dataset.mat')
train_data = ds['Train_Data']
test_data = ds['Test_Data']


train_set = np.array([x[0:51] for x in train_data])
test_Set = np.array([x[0:51] for x in test_data])

y_true_train = retrive_label(train_data)
y_true_test = retrive_label(test_data)

gnb = GaussianNB()
y_pred = gnb.fit(train_set, y_true_train).predict(test_Set)

print(accuracy_score(y_true_test, y_pred))
