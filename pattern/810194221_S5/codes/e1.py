import numpy as np
import scipy.io
from scipy.stats import multivariate_normal

ds = scipy.io.loadmat('dataset.mat')
train_data = ds['Train_Data']
test_data = ds['Test_Data']


train_data_w1 = np.array([x[0:51] for x in train_data if x[51] == 1])
train_data_w2 = np.array([x[0:51] for x in train_data if x[52] == 1])
train_data_w3 = np.array([x[0:51] for x in train_data if x[53] == 1])
train_data_w4 = np.array([x[0:51] for x in train_data if x[54] == 1])
train_data_w5 = np.array([x[0:51] for x in train_data if x[55] == 1])

temp = np.array([train_data_w1,train_data_w2,train_data_w3,train_data_w4,train_data_w5])

mu_hat = []
cov_hat = []
p_w = []
all_n = train_data.shape[0]

for item in temp:
    n = item.shape[0]
    p_w.append(n/all_n)
    mu_hat.append(item.sum(axis=0)/n)
    cov_hat.append(np.cov(item.T))


labels = []

true_positive = np.zeros(5)
false_positive = np.zeros(5)
true_negative = np.zeros(5)
false_negative = np.zeros(5)
conf_matrix = np.zeros((5, 5))

for item in test_data:
    label = 0
    max_p = 0
    for i in range(0, 5):
        p = multivariate_normal.pdf(item[0:51], mean=mu_hat[i], cov=cov_hat[i], allow_singular=True) * p_w[i]
        if p > max_p:
            max_p = p
            label = i

    labels.append(label)
    correct_label = np.where(item[51:] == 1)[0][0]
    conf_matrix[label][correct_label] += 1
    if label == correct_label:
        true_positive[label] += 1
        true_negative[:] += 1
        true_negative[label] -= 1
    else:
        false_positive[label] += 1
        false_negative[correct_label] += 1
        true_negative[:] += 1
        true_negative[label] -= 1
        true_negative[label] -= 1

conf_matrix = conf_matrix / conf_matrix.sum(axis=0)
accr = np.diag(conf_matrix).sum() / float(conf_matrix.sum())
print("Average Correct Classification Rate is {0}".format(accr))
print(conf_matrix)

for i in range(0, 5):
    print("precision of w{0} : {1}".format(i+1, true_positive[i]/(true_positive[i]+false_positive[i])))
    print("recall of w{0} : {1}".format(i + 1, true_positive[i] / (true_positive[i] + false_negative[i])))










