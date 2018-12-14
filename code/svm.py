import scipy.io as sio
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score

np.random.seed(0)


data = sio.loadmat('../data/hand_written_digits_545.mat')
X = data['training_data']
y = data['training_label']
idx = np.array([l for l in range(len(y)) if y[l] == 4 or y[l] == 9])
print(idx)
X = X[idx]
y = y[idx]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# clf = svm.SVC(kernel = 'rbf', gamma = 1, C = 0.7)
clf = svm.SVC(kernel = 'linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

p = precision_score(y_test, y_pred, average='micro')  
print(p)import numpy as np 
import random 
