
import tests as tests
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

#############################################################
'''Acknowledgement: the Data and SupportVectorMachine classes are reused from homework 4 of CS6242 DVA'''


class Data():

    def dataAllocation(self, path):
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

    def trainSets(self, x_data, y_data):
        # Split 70% of the data into training and 30% into test sets. Call them x_train, x_test, y_train and y_test.
        # Use the train_test_split method in sklearn with the parameter 'shuffle' set to true and the 'random_state' set to 614.
        # args: pandas dataframe, pandas dataframe
        # return: pandas dataframe, pandas dataframe, pandas series, pandas series
        # -------------------------------
        # ADD CODE HERE
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=0.2, shuffle=True, random_state=614)
        # -------------------------------
        return x_train, x_test, y_train, y_test


##################################################
##### Do not add anything below this line ########
tests.dataTest(Data)
##################################################


class DecisionTrees():

    def DTsClassifier(self, x_train, x_test, y_train):
        # Create a DecisionForestClassifier and train it.
        # args: pandas dataframe, pandas dataframe, numpy array
        # return: DTsClassifier object, numpy array, numpy array
        '''DEFAULT: class sklearn.tree.DecisionTreeClassifier(*, criterion='gini', splitter='best',
        max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
        max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
        min_impurity_split=None, class_weight=None, ccp_alpha=0.0)'''
        # try criterion = 'entropy', splitter='best' or 'random', max_depth = 3~5, min_samples_leaf = 3~5
        # -------------------------------
        dt_clf = tree.DecisionTreeClassifier(
            criterion='entropy', splitter='best', max_depth=4, ccp_alpha=0.02, random_state=0)
        dt_clf.fit(x_train, y_train)
        y_predict_train = dt_clf.predict(x_train)
        y_predict_test = dt_clf.predict(x_test)
        # -------------------------------
        return dt_clf, y_predict_train, y_predict_test

    def DTsTrainAccuracy(self, y_train, y_predict_train):
        # Return accuracy on the training set using the accuracy_score method.
        # args: pandas series, numpy array
        # return: float
        # -------------------------------
        # ADD CODE HERE
        train_accuracy = accuracy_score(y_train, y_predict_train)
        # -------------------------------
        return train_accuracy

    def DTsTestAccuracy(self, y_test, y_predict_test):
        # Return accuracy on the test set using the accuracy_score method.
        # args: pandas series, numpy array
        # return: float
        # -------------------------------
        # ADD CODE HERE
        test_accuracy = accuracy_score(y_test, y_predict_test)
        # -------------------------------
        return test_accuracy

    def DTsFeatureImportance(self, dt_clf):
        # Determine the feature importance as evaluated by the Decision Tree Classifier.
        # args: DTsClassifier object
        # return: sorted array of importance in ascending order
        # -------------------------------
        # ADD CODE HERE
        feature_importance = dt_clf.feature_importances_
        # -------------------------------
        return np.argsort(feature_importance)

    def ReceiverOperatingCharacteristic(self, dt_clf, x_test, y_test):
        # Plot ROC curve.
        # args: pandas dataframe, numpy array
        # -------------------------------
        # ADD CODE HERE
        dt_curve = plot_roc_curve(dt_clf, x_test, y_test)
        # plot_roc_curve(dt_clf, x_test, y_test, ax=dt_curve.ax_)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.show()

    def hyperParameterTuning(self, x_train, x_test, y_train):
        # Tune the hyper-parameters and 'max_depth' and 'min_samples_leaf'
        # args: pandas dataframe, pandas dataframe, numpy array
        # return: best_params dict, numpy array, numpy array
        # 'max_depth': [2, 8, 16]
        # 'min_samples_leaf': [2, 8, 16]
        # -------------------------------
        # ADD CODE HERE
        dt_clf = tree.DecisionTreeClassifier(
            criterion='entropy', random_state=0)
        param_grid = {'max_depth': [
            2, 4, 6, 8, 10], 'ccp_alpha': [0.02, 0.04, 0.06, 0.08, 0.1]}
        gscv_dt = GridSearchCV(estimator=dt_clf, param_grid=param_grid, cv=10)
        gscv_dt.fit(x_train, y_train)
        y_predict_train = gscv_dt.predict(x_train)
        y_predict_test = gscv_dt.predict(x_test)
        # -------------------------------
        return gscv_dt.best_params_, y_predict_train, y_predict_test


##################################################
##### Test the classifier ########
#tests.DecisionTreesTest(Data, DecisionTrees)
##################################################


class NeuralNetwork():

    def dataPreProcess(self, x_train, x_test):
        # Pre-process the data to standardize it, otherwise the grid search will take much longer.
        # args: pandas dataframe, pandas dataframe
        # return: pandas dataframe, pandas dataframe
        # -------------------------------
        # ADD CODE HERE
        scaler = StandardScaler()
        scaler.fit(x_train)
        scaled_x_train = scaler.transform(x_train)
        scaled_x_test = scaler.transform(x_test)
        # -------------------------------
        return scaled_x_train, scaled_x_test

    def NNClassifier(self, x_train, x_test, y_train):
        # Create a DecisionForestClassifier and train it.
        # args: pandas dataframe, pandas dataframe, numpy array
        # return: DTsClassifier object, numpy array, numpy array
        '''DEFAULT: class sklearn.neural_network.MLPClassifier(hidden_layer_sizes=100, activation='relu',
        *, solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001,
        power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False,
        momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9,
        beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)[source]¶'''
        # try hidden_layer_sizes = (10，10，10), activation='logistic' or 'relu', learning_rate_init = 0.01
        # -------------------------------
        nn_clf = MLPClassifier(
            hidden_layer_sizes=(3), activation='logistic', solver='sgd', learning_rate_init=0.05, max_iter=10000, random_state=0)
        nn_clf.fit(x_train, y_train)
        y_predict_train = nn_clf.predict(x_train)
        y_predict_test = nn_clf.predict(x_test)
        # -------------------------------
        return nn_clf, y_predict_train, y_predict_test

    def NNTrainAccuracy(self, y_train, y_predict_train):
        # Return accuracy on the training set using the accuracy_score method.
        # args: pandas series, numpy array
        # return: float
        # -------------------------------
        # ADD CODE HERE
        train_accuracy = accuracy_score(y_train, y_predict_train)
        # -------------------------------
        return train_accuracy

    def NNTestAccuracy(self, y_test, y_predict_test):
        # Return accuracy on the test set using the accuracy_score method.
        # args: pandas series, numpy array
        # return: float
        # -------------------------------
        # ADD CODE HERE
        test_accuracy = accuracy_score(y_test, y_predict_test)
        # -------------------------------
        return test_accuracy

    def ReceiverOperatingCharacteristic(self, nn_clf, x_test, y_test):
        # Plot ROC curve.
        # args: pandas dataframe, numpy array
        # -------------------------------
        # ADD CODE HERE
        nn_curve = plot_roc_curve(nn_clf, x_test, y_test)
        # plot_roc_curve(dt_clf, x_test, y_test, ax=dt_curve.ax_)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.show()

    def hyperParameterTuning(self, x_train, x_test, y_train):
        # Tune the hyper-parameters and 'max_depth' and 'min_samples_leaf'
        # args: pandas dataframe, pandas dataframe, numpy array
        # return: best_params dict, numpy array, numpy array
        # 'hidden_layer_sizes': [(10，10), (10，10，10, 10), (10，10，10, 10, 10, 10, 10, 10)]
        # 'activation': ['relu', 'logistic']
        # 'learning_rate_init': [0.001, 0.005, 0.025]
        # -------------------------------
        # ADD CODE HERE
        nn_clf = MLPClassifier(max_iter=10000, random_state=0)
        param_grid = {'hidden_layer_sizes': [(2), (3), (4), (5), (6), (7), (8)], 'activation': [
            'logistic'], 'learning_rate_init': [0.02, 0.04, 0.06, 0.08, 0.1]}
        gscv_nn = GridSearchCV(estimator=nn_clf, param_grid=param_grid, cv=10)
        gscv_nn.fit(x_train, y_train)
        y_predict_train = gscv_nn.predict(x_train)
        y_predict_test = gscv_nn.predict(x_test)
        # -------------------------------
        return gscv_nn.best_params_, y_predict_train, y_predict_test


##################################################
##### Test the classifier ########
#tests.NeuralNetworkTest(Data, NeuralNetwork)
##################################################


class GradientBoostingDecisionTrees():

    def GBDTsClassifier(self, x_train, x_test, y_train):
        # Create a GradientBoostingClassifier and train it.
        # args: pandas dataframe, pandas dataframe, numpy array
        # return: GBDTsClassifier object, numpy array, numpy array
        '''DEFAULT: class sklearn.ensemble.GradientBoostingClassifier(*, loss='deviance', learning_rate=0.1,
        n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1,
        min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None,
        init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False,
        validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)'''
        # try criterion = 'entropy', splitter='best' or 'random', max_depth = 3~5, min_samples_leaf = 3~5
        # -------------------------------
        gbdt_clf = GradientBoostingClassifier(
            n_estimators=15, max_depth=2, random_state=0)
        gbdt_clf.fit(x_train, y_train)
        y_predict_train = gbdt_clf.predict(x_train)
        y_predict_test = gbdt_clf.predict(x_test)
        # -------------------------------
        return gbdt_clf, y_predict_train, y_predict_test

    def GBDTsTrainAccuracy(self, y_train, y_predict_train):
        # Return accuracy on the training set using the accuracy_score method.
        # args: pandas series, numpy array
        # return: float
        # -------------------------------
        # ADD CODE HERE
        train_accuracy = accuracy_score(y_train, y_predict_train)
        # -------------------------------
        return train_accuracy

    def GBDTsTestAccuracy(self, y_test, y_predict_test):
        # Return accuracy on the test set using the accuracy_score method.
        # args: pandas series, numpy array
        # return: float
        # -------------------------------
        # ADD CODE HERE
        test_accuracy = accuracy_score(y_test, y_predict_test)
        # -------------------------------
        return test_accuracy

    def GBDTFeatureImportance(self, gbdt_clf):
        # Determine the feature importance as evaluated by the Decision Tree Classifier.
        # args: DTsClassifier object
        # return: sorted array of importance in ascending order
        # -------------------------------
        # ADD CODE HERE
        feature_importance = gbdt_clf.feature_importances_
        # -------------------------------
        return np.argsort(feature_importance)

    def ReceiverOperatingCharacteristic(self, gbdt_clf, x_test, y_test):
        # Plot ROC curve.
        # args: pandas dataframe, numpy array
        # -------------------------------
        # ADD CODE HERE
        gbdt_curve = plot_roc_curve(gbdt_clf, x_test, y_test)
        # plot_roc_curve(dt_clf, x_test, y_test, ax=dt_curve.ax_)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.show()

    def hyperParameterTuning(self, x_train, x_test, y_train):
        # Tune the hyper-parameters and 'max_depth' and 'min_samples_leaf'
        # args: pandas dataframe, pandas dataframe, numpy array
        # return: best_params dict, numpy array, numpy array
        # 'max_depth': [2, 8, 16]
        # 'n_estimators': [20, 40, 80]
        # -------------------------------
        # ADD CODE HERE
        gbdt_clf = GradientBoostingClassifier(random_state=0)
        param_grid = {'max_depth': [
            2], 'n_estimators': [10, 15, 20, 25, 30]}
        gscv_gbdt = GridSearchCV(
            estimator=gbdt_clf, param_grid=param_grid, cv=10)
        gscv_gbdt.fit(x_train, y_train)
        y_predict_train = gscv_gbdt.predict(x_train)
        y_predict_test = gscv_gbdt.predict(x_test)
        # -------------------------------
        return gscv_gbdt.best_params_, y_predict_train, y_predict_test


##################################################
##### Test the classifier ########
#tests.GradientBoostingDecisionTreesTest(Data, GradientBoostingDecisionTrees)
##################################################

class SupportVectorMachine():

    def dataPreProcess(self, x_train, x_test):
        # Pre-process the data to standardize it, otherwise the grid search will take much longer.
        # args: pandas dataframe, pandas dataframe
        # return: pandas dataframe, pandas dataframe
        # -------------------------------
        # ADD CODE HERE
        scaler = StandardScaler()
        scaler.fit(x_train)
        scaled_x_train = scaler.transform(x_train)
        scaled_x_test = scaler.transform(x_test)
        # -------------------------------
        return scaled_x_train, scaled_x_test

    def SVCClassifier(self, scaled_x_train, scaled_x_test, y_train):
        # Create a SVC classifier and train it. Set gamma = 'auto'
        # args: pandas dataframe, pandas dataframe, pandas series
        # return: numpy array, numpy array
        # -------------------------------
        '''DEFAULT: class sklearn.svm.SVC(*, C=1.0, kernel='rbf', degree=3, gamma='scale', 
        coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, 
        verbose=False, max_iter=- 1, decision_function_shape='ovr', break_ties=False, random_state=None)'''
        # kernel = 'rbf', 'sigmoid', C = 0.2, 1.0, 5.0
        # ADD CODE HERE
        svm_clf = SVC(C=0.01, kernel='linear', gamma='auto', random_state=0)
        svm_clf.fit(scaled_x_train, y_train)
        y_predict_train = svm_clf.predict(scaled_x_train)
        y_predict_test = svm_clf.predict(scaled_x_test)
        # -------------------------------
        return svm_clf, y_predict_train, y_predict_test

    def SVCTrainAccuracy(self, y_train, y_predict_train):
        # Return accuracy on the training set using the accuracy_score method.
        # args: pandas series, numpy array
        # return: float
        # -------------------------------
        # ADD CODE HERE
        train_accuracy = accuracy_score(y_train, y_predict_train)
        # -------------------------------
        return train_accuracy

    def SVCTestAccuracy(self, y_test, y_predict_test):
        # Return accuracy on the test set using the accuracy_score method.
        # args: pandas series, numpy array
        # return: float
        # -------------------------------
        # ADD CODE HERE
        test_accuracy = accuracy_score(y_test, y_predict_test)
        # -------------------------------
        return test_accuracy

    def ReceiverOperatingCharacteristic(self, svm_clf, x_test, y_test):
        # Plot ROC curve.
        # args: pandas dataframe, numpy array
        # -------------------------------
        # ADD CODE HERE
        svm_curve = plot_roc_curve(svm_clf, x_test, y_test)
        # plot_roc_curve(dt_clf, x_test, y_test, ax=dt_curve.ax_)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.show()

    def hyperParameterTuning(self, x_train, x_test, y_train):
        # Tune the hyper-parameters
        # args: pandas dataframe, pandas dataframe, numpy array
        # return: best_params dict, numpy array, numpy array
        # 'kernel': 'linear', 'rbf', 'sigmoid'
        # 'C': [0.02, 1.0, 5.0]
        # -------------------------------
        # ADD CODE HERE
        svm_clf = SVC(gamma="auto", random_state=0)
        param_grid = {'kernel': ('linear', 'poly', 'rbf', 'sigmoid'), 'C': np.linspace(0.01, 0.1, 10)}
        gscv_svm = GridSearchCV(
            estimator=svm_clf, param_grid=param_grid, cv=10)
        gscv_svm.fit(x_train, y_train)
        y_predict_train = gscv_svm.predict(x_train)
        y_predict_test = gscv_svm.predict(x_test)
        # -------------------------------
        return gscv_svm.best_params_, y_predict_train, y_predict_test


##################################################
##### Do not add anything below this line ########
#tests.SupportVectorMachineTest(Data, SupportVectorMachine)
##################################################


class kNearestNeighbors():

    def dataPreProcess(self, x_train, x_test):
        # Pre-process the data to standardize it, otherwise the grid search will take much longer.
        # args: pandas dataframe, pandas dataframe
        # return: pandas dataframe, pandas dataframe
        # -------------------------------
        # ADD CODE HERE
        scaler = StandardScaler()
        scaler.fit(x_train)
        scaled_x_train = scaler.transform(x_train)
        scaled_x_test = scaler.transform(x_test)
        # -------------------------------
        return scaled_x_train, scaled_x_test

    def kNNClassifier(self, scaled_x_train, scaled_x_test, y_train):
        # Create a kNN classifier and train it.
        # args: pandas dataframe, pandas dataframe, pandas series
        # return: numpy array, numpy array
        # -------------------------------
        '''DEFAULT: class sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, *, weights='uniform', 
        algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None, **kwargs)'''
        # n_neighbors = 5, weights = 'uniform' or 'distance', p = 1 (Manhattan), 2 (Euclidean)
        # ADD CODE HERE
        knn_clf = KNeighborsClassifier(
            n_neighbors=17, p=2)
        knn_clf.fit(scaled_x_train, y_train)
        y_predict_train = knn_clf.predict(scaled_x_train)
        y_predict_test = knn_clf.predict(scaled_x_test)
        # -------------------------------
        return knn_clf, y_predict_train, y_predict_test

    def KNNTrainAccuracy(self, y_train, y_predict_train):
        # Return accuracy on the training set using the accuracy_score method.
        # args: pandas series, numpy array
        # return: float
        # -------------------------------
        # ADD CODE HERE
        train_accuracy = accuracy_score(y_train, y_predict_train)
        # -------------------------------
        return train_accuracy

    def KNNTestAccuracy(self, y_test, y_predict_test):
        # Return accuracy on the test set using the accuracy_score method.
        # args: pandas series, numpy array
        # return: float
        # -------------------------------
        # ADD CODE HERE
        test_accuracy = accuracy_score(y_test, y_predict_test)
        # -------------------------------
        return test_accuracy

    def ReceiverOperatingCharacteristic(self, knn_clf, x_test, y_test):
        # Plot ROC curve.
        # args: pandas dataframe, numpy array
        # -------------------------------
        # ADD CODE HERE
        knn_curve = plot_roc_curve(knn_clf, x_test, y_test)
        # plot_roc_curve(dt_clf, x_test, y_test, ax=dt_curve.ax_)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.show()

    def hyperParameterTuning(self, x_train, x_test, y_train):
        # Tune the hyper-parameters
        # args: pandas dataframe, pandas dataframe, numpy array
        # return: best_params dict, numpy array, numpy array
        # n_neighbors = 5,
        # weights = 'uniform' or 'distance'
        # p = 1 (Manhattan), 2 (Euclidean)
        # -------------------------------
        # ADD CODE HERE
        knn_clf = KNeighborsClassifier()
        param_grid = {'n_neighbors': np.linspace(12, 20, 9,dtype=int), 'p': [2]}
        gscv_knn = GridSearchCV(
            estimator=knn_clf, param_grid=param_grid, cv=10)
        gscv_knn.fit(x_train, y_train)
        y_predict_train = gscv_knn.predict(x_train)
        y_predict_test = gscv_knn.predict(x_test)
        # -------------------------------
        return gscv_knn.best_params_, y_predict_train, y_predict_test


##################################################
##### Do not add anything below this line ########
tests.kNearestNeighborsTest(Data, kNearestNeighbors)
##################################################
