
import numpy as np
import pandas as pd
import time
import gc
import random
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn import tree
from sklearn.metrics import plot_roc_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import ShuffleSplit

'''The plot_validation_curve method is modified based on:
https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html#sphx-glr-auto-examples-model-selection-plot-validation-curve-py
'''

def dataAllocation(path):
    # Separate out the x_data and y_data and return each
    # args: string path for .csv file
    # return: pandas dataframe, pandas series
    # -------------------------------
    # ADD CODE HERE
    df = pd.read_csv(path)
    xcols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']
    ycol = ['y']
    x_data = df[xcols]
    y_data = df[ycol]
#        print(y_data[y_data.y == 1].shape[0])
#       print(df.shape[0])
    # -------------------------------
    return x_data, y_data.values.ravel()


x_data, y_data = dataAllocation(path='data/pima-indians-diabetes.csv')
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

""" # DTs
estimator = tree.DecisionTreeClassifier(
    criterion='entropy', splitter='best', max_depth = 6, random_state=0)
param_range = np.linspace(0, 0.1, 100)
train_scores, test_scores = validation_curve(
    estimator, x_data, y_data, param_name="ccp_alpha", param_range=param_range,
    scoring="accuracy", n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

axes[0].set_xlabel(r"ccp_alpha")
axes[0].set_ylabel("Score")
axes[0].grid()
axes[0].set_ylim(0.5, 1.0)
lw = 2
axes[0].plot(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
axes[0].fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
axes[0].plot(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
axes[0].fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
axes[0].legend(loc="best")
axes[0].set_title("Validation curve with max_depth=6")


estimator = tree.DecisionTreeClassifier(
    criterion='entropy', splitter='best', ccp_alpha = 0.01, random_state=0)
param_range = np.linspace(2, 20, 10)
train_scores, test_scores = validation_curve(
    estimator, x_data, y_data, param_name="max_depth", param_range=param_range,
    scoring="accuracy", n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

axes[1].set_xlabel(r"max_depth")
axes[1].set_ylabel("Score")
axes[1].set_ylim(0.5, 1.0)
axes[1].grid()
lw = 2
axes[1].plot(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
axes[1].fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
axes[1].plot(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
axes[1].fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
axes[1].legend(loc="best")
axes[1].set_title("Validation curve with ccp_alpha=0.01") """


""" # NN
scaler = StandardScaler()
scaler.fit(x_data)
x_data = scaler.transform(x_data)
estimator = MLPClassifier(
            hidden_layer_sizes=(3), activation='logistic', solver='sgd', max_iter=10000, random_state=0)

param_range = np.linspace(0.005, 0.1, 20)
train_scores, test_scores = validation_curve(
    estimator, x_data, y_data, param_name="learning_rate_init", param_range=param_range,
    scoring="accuracy", n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

axes[0].set_xlabel(r"learning_rate_init")
axes[0].set_ylabel("Score")
axes[0].grid()
axes[0].set_ylim(0.5, 1.0)
lw = 2
axes[0].plot(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
axes[0].fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
axes[0].plot(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
axes[0].fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
axes[0].legend(loc="best")
axes[0].set_title("Validation curve with hidden_layer_sizes=(3)")

estimator = MLPClassifier(
            learning_rate_init=0.005, activation='logistic', solver='sgd', max_iter=10000, random_state=0)
param_range = [(3),(6),(9),(3,3),(6,6),(9,9)]
train_scores, test_scores = validation_curve(
    estimator, x_data, y_data, param_name="hidden_layer_sizes", param_range=param_range,
    scoring="accuracy", n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

axes[1].set_xlabel(r"hidden_layer_sizes")
axes[1].set_ylabel("Score")
axes[1].set_ylim(0.5, 1.0)
axes[1].grid()
lw = 2
axes[1].plot([1,2,3,4,5,6], train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
axes[1].fill_between([1,2,3,4,5,6], train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
axes[1].plot([1,2,3,4,5,6], test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
axes[1].fill_between([1,2,3,4,5,6], test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
axes[1].legend(loc="best")
axes[1].set_title("Validation curve with learning_rate_init=0.005") """


""" # GBDTs
estimator = GradientBoostingClassifier(
            n_estimators=20, random_state=0)

param_range = np.linspace(2, 8, 7)
train_scores, test_scores = validation_curve(
estimator, x_data, y_data, param_name="max_depth", param_range=param_range,
    scoring="accuracy", n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

axes[0].set_xlabel(r"max_depth")
axes[0].set_ylabel("Score")
axes[0].grid()
axes[0].set_ylim(0.5, 1.0)
lw = 2
axes[0].plot(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
axes[0].fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
axes[0].plot(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
axes[0].fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
axes[0].legend(loc="best")
axes[0].set_title("Validation curve with n_estimator=20")

estimator = GradientBoostingClassifier(
            max_depth=2, random_state=0)
param_range = [5, 10, 15, 20, 25, 30, 35, 40]
train_scores, test_scores = validation_curve(
estimator, x_data, y_data, param_name="n_estimators", param_range=param_range,
    scoring="accuracy", n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

axes[1].set_xlabel(r"n_estimators")
axes[1].set_ylabel("Score")
axes[1].set_ylim(0.5, 1.0)
axes[1].grid()
lw = 2
axes[1].plot(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
axes[1].fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
axes[1].plot(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
axes[1].fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
axes[1].legend(loc="best")
axes[1].set_title("Validation curve with max_depth=2") """


""" # SVM
scaler = StandardScaler()
scaler.fit(x_data)
x_data = scaler.transform(x_data)
estimator = SVC(kernel='linear', gamma='auto', random_state=0)

param_range = np.linspace(0.02, 1, 50)
train_scores, test_scores = validation_curve(
    estimator, x_data, y_data, param_name="C", param_range=param_range,
    scoring="accuracy", n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

axes[0].set_xlabel(r"C")
axes[0].set_ylabel("Score")
axes[0].grid()
axes[0].set_ylim(0.5, 1.0)
lw = 2
axes[0].plot(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
axes[0].fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
axes[0].plot(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
axes[0].fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
axes[0].legend(loc="best")
axes[0].set_title("Validation curve with kernel=linear")

estimator = SVC(C=0.01, gamma='auto', random_state=0)
param_range = ['linear', 'poly', 'rbf', 'sigmoid']
train_scores, test_scores = validation_curve(
    estimator, x_data, y_data, param_name="kernel", param_range=param_range,
    scoring="accuracy", n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

axes[1].set_xlabel(r"kernel")
axes[1].set_ylabel("Score")
axes[1].set_ylim(0.5, 1.0)
axes[1].grid()
lw = 2
axes[1].plot([1,2,3,4], train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
axes[1].fill_between([1,2,3,4], train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
axes[1].plot([1,2,3,4], test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
axes[1].fill_between([1,2,3,4], test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
axes[1].legend(loc="best")
axes[1].set_title("Validation curve with C=0.01") """


# kNN
scaler = StandardScaler()
scaler.fit(x_data)
x_data = scaler.transform(x_data)
estimator = KNeighborsClassifier(p=2)

param_range = np.linspace(2, 20, 19,dtype=int)
train_scores, test_scores = validation_curve(
    estimator, x_data, y_data, param_name="n_neighbors", param_range=param_range,
    scoring="accuracy", n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

axes[0].set_xlabel(r"n_neighbors")
axes[0].set_ylabel("Score")
axes[0].grid()
axes[0].set_ylim(0.5, 1.0)
lw = 2
axes[0].plot(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
axes[0].fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
axes[0].plot(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
axes[0].fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
axes[0].legend(loc="best")
axes[0].set_title("Validation curve with p=2")

estimator = KNeighborsClassifier(n_neighbors=6)
param_range = [1, 2]
train_scores, test_scores = validation_curve(
    estimator, x_data, y_data, param_name="p", param_range=param_range,
    scoring="accuracy", n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

axes[1].set_xlabel(r"distance metric")
axes[1].set_ylabel("Score")
axes[1].set_ylim(0.5, 1.0)
axes[1].grid()
lw = 2
axes[1].plot(param_range , train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
axes[1].fill_between(param_range , train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
axes[1].plot(param_range , test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
axes[1].fill_between(param_range , test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
axes[1].legend(loc="best")
axes[1].set_title("Validation curve with n_neighbors=6")

plt.show()
