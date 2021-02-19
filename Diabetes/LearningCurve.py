
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
from sklearn.model_selection import ShuffleSplit

'''The plot_learning_curve method is modified based on:
 https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py'''


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


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot fit_time vs score
    axes[1].grid()
    axes[1].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[1].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[1].set_xlabel("fit_times")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Performance of the model")

    return plt



x_data, y_data = dataAllocation(path='data/pima-indians-diabetes.csv')

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

# DTs
""" estimator = tree.DecisionTreeClassifier(
    criterion='entropy', splitter='best', max_depth=5, ccp_alpha=0.01, random_state=0) """
# NN
""" scaler = StandardScaler()
scaler.fit(x_data)
x_data = scaler.transform(x_data)
estimator = MLPClassifier(
            hidden_layer_sizes=(3), activation='logistic', solver='sgd', learning_rate_init=0.005, max_iter=10000, random_state=0) """

# GBDTs
""" estimator = GradientBoostingClassifier(
            n_estimators=20, max_depth=2, random_state=0) """

# SVM
""" scaler = StandardScaler()
scaler.fit(x_data)
x_data = scaler.transform(x_data)
estimator = SVC(C=0.1, kernel='linear', gamma='auto', random_state=0) """

# kNN
scaler = StandardScaler()
scaler.fit(x_data)
x_data = scaler.transform(x_data)
estimator = KNeighborsClassifier(n_neighbors=6, weights='uniform', p=2)

train_sizes, train_scores, test_scores, fit_times, _ = \
    learning_curve(estimator, x_data, y_data, cv=cv, n_jobs=4,
                    train_sizes=np.linspace(.1, 1.0, 5),
                    return_times=True)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
fit_times_mean = np.mean(fit_times, axis=1)
fit_times_std = np.std(fit_times, axis=1)

axes[0].set_xlabel("Numer of training samples")
axes[0].set_ylabel("Score")
axes[0].grid()
axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color="r")
axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1,
                        color="g")
axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                label="Training score")
axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                label="Cross-validation score")
axes[0].legend(loc="best")
axes[0].set_title("Learning curve")

# Plot fit_time vs score
axes[1].grid()
axes[1].plot(fit_times_mean, test_scores_mean, 'o-')
axes[1].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1)
axes[1].set_xlabel("fit_times")
axes[1].set_ylabel("Score")
axes[1].set_title("Model performance")

plt.show()

""" title = "Learning Curves (Naive Bayes)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator = tree.DecisionTreeClassifier(
    criterion='entropy', splitter='best', max_depth=5, ccp_alpha=0.01, random_state=0)
plot_learning_curve(estimator, title, x_data, y_data, axes=axes[:, 0], ylim=(0.5, 1.0),
                    cv=cv, n_jobs=4)

title = r"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
estimator = tree.DecisionTreeClassifier(
    criterion='entropy', splitter='best', max_depth=5, ccp_alpha=0.01, random_state=0)
plot_learning_curve(estimator, title, x_data, y_data, axes=axes[:, 1], ylim=(0.5, 1.0),
                    cv=cv, n_jobs=4)

plt.show() """
