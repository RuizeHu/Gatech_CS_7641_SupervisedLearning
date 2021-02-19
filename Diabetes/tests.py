import platform
import random
import pandas as pd
import time

if platform.system() != 'Windows':
    import resource


def dataTest(Data):
    datatest = Data()
    data = 'data/pima-indians-diabetes.csv'
    try:
        x_data, y_data = datatest.dataAllocation(data)
        print("dataAllocation Function Executed")
    except:
        print("Data not imported correctly")
    try:
        x_train, x_test, y_train, y_test = datatest.trainSets(x_data, y_data)
        print("trainSets Function Executed")
    except:
        print("Data not imported correctly")


def DecisionTreesTest(Data, DecisionTrees):
    dataset = Data()
    dt = DecisionTrees()
    data = 'data/pima-indians-diabetes.csv'
    x_data, y_data = dataset.dataAllocation(data)
    x_train, x_test, y_train, y_test = dataset.trainSets(x_data, y_data)
    try:
        t0 = time.time()
        dt_clf, y_predict_train, y_predict_test = dt.DTsClassifier(
            x_train, x_test, y_train)
        print("seconds wall time:", time.time() - t0)
        print("DecisionTree Function Executed")
    except:
        print("Failed to execute DecisionTree()")
    try:
        print("Decision Tree Train Accuracy: ",
              dt.DTsTrainAccuracy(y_train, y_predict_train))
    except:
        print("Failed to execute dtTrainAccuracy()")
    try:
        print("Decision Tree Test Accuracy: ",
              dt.DTsTestAccuracy(y_test, y_predict_test))
    except:
        print("Failed to execute dtTrainAccuracy()")
    try:
        print("Decision Tree Feature Importance: ",
              dt.DTsFeatureImportance(dt_clf))
    except:
        print("Failed to execute DTsFeatureImportance()")
    try:
        print("Plot_ROC_Curve Function Executed",
              dt.ReceiverOperatingCharacteristic(dt_clf, x_test, y_test))
    except:
        print("Failed to Plot_ROC_Curve")
    try:
        best_params, y_predict_train, y_predict_test = dt.hyperParameterTuning(
            x_train, x_test, y_train)
        print("HyperParameterTuning Function Executed")
        print("Best Parameters:", best_params)
        print("Decision Tree Train Accuracy: ",
              dt.DTsTrainAccuracy(y_train, y_predict_train))
        print("Decision Tree Test Accuracy: ",
              dt.DTsTestAccuracy(y_test, y_predict_test))
    except:
        print("Failed to execute hyperParameterTuning()")


def NeuralNetworkTest(Data, NeuralNetwork):
    dataset = Data()
    nn = NeuralNetwork()
    data = 'data/pima-indians-diabetes.csv'
    x_data, y_data = dataset.dataAllocation(data)
    x_train, x_test, y_train, y_test = dataset.trainSets(x_data, y_data)
    try:
        x_train, x_test = nn.dataPreProcess(x_train, x_test)
        print("dataPreProcess Function Executed")
    except:
        print("Failed to execute dataPreProcess()")
    try:
        t0 = time.time()
        nn_clf, y_predict_train, y_predict_test = nn.NNClassifier(
            x_train, x_test, y_train)
        print("seconds wall time:", time.time() - t0)
        print("NeuralNetwork Function Executed")
    except:
        print("Failed to execute NeuralNetwork()")
    try:
        print("Neural Network Train Accuracy: ",
              nn.NNTrainAccuracy(y_train, y_predict_train))
    except:
        print("Failed to execute NNTrainAccuracy()")
    try:
        print("Neural Network Test Accuracy: ",
              nn.NNTestAccuracy(y_test, y_predict_test))
    except:
        print("Failed to execute nnTrainAccuracy()")
    try:
        print("Plot_ROC_Curve Function Executed",
              nn.ReceiverOperatingCharacteristic(nn_clf, x_test, y_test))
    except:
        print("Failed to Plot_ROC_Curve")
    try:
        best_params, y_predict_train, y_predict_test = nn.hyperParameterTuning(
            x_train, x_test, y_train)
        print("HyperParameterTuning Function Executed")
        print("Best Parameters:", best_params)
        print("Neural Network Train Accuracy: ",
              nn.NNTrainAccuracy(y_train, y_predict_train))
        print("Neural Network Test Accuracy: ",
              nn.NNTestAccuracy(y_test, y_predict_test))
    except:
        print("Failed to execute hyperParameterTuning()")


def GradientBoostingDecisionTreesTest(Data, GradientBoostingDecisionTrees):
    dataset = Data()
    gbdt = GradientBoostingDecisionTrees()
    data = 'data/pima-indians-diabetes.csv'
    x_data, y_data = dataset.dataAllocation(data)
    x_train, x_test, y_train, y_test = dataset.trainSets(x_data, y_data)
    try:
        t0 = time.time()
        gbdt_clf, y_predict_train, y_predict_test = gbdt.GBDTsClassifier(
            x_train, x_test, y_train)
        print("seconds wall time:", time.time() - t0)
        print("GradientBoostingDecisionTree Function Executed")
    except:
        print("Failed to execute GradientBoostingDecisionTree()")
    try:
        print("Gradient Boosting Decision Tree Train Accuracy: ",
              gbdt.GBDTsTrainAccuracy(y_train, y_predict_train))
    except:
        print("Failed to execute gbdtTrainAccuracy()")
    try:
        print("Gradient Boosting Decision Tree Test Accuracy: ",
              gbdt.GBDTsTestAccuracy(y_test, y_predict_test))
    except:
        print("Failed to execute gbdtTrainAccuracy()")
    try:
        print("Decision Tree Feature Importance: ",
              gbdt.GBDTFeatureImportance(gbdt_clf))
    except:
        print("Failed to execute NNFeatureImportance()")
    try:
        print("Plot_ROC_Curve Function Executed",
              gbdt.ReceiverOperatingCharacteristic(gbdt_clf, x_test, y_test))
    except:
        print("Failed to Plot_ROC_Curve")
    try:
        best_params, y_predict_train, y_predict_test = gbdt.hyperParameterTuning(
            x_train, x_test, y_train)
        print("HyperParameterTuning Function Executed")
        print("Best Parameters:", best_params)
        print("Gradient Boosting Decision Tree Train Accuracy: ",
              gbdt.GBDTsTrainAccuracy(y_train, y_predict_train))
        print("Gradient Boosting Decision Tree Test Accuracy: ",
              gbdt.GBDTsTestAccuracy(y_test, y_predict_test))
    except:
        print("Failed to execute hyperParameterTuning()")


def SupportVectorMachineTest(Data, SupportVectorMachine):
    dataset = Data()
    svm = SupportVectorMachine()
    data = 'data/pima-indians-diabetes.csv'
    x_data, y_data = dataset.dataAllocation(data)
    x_train, x_test, y_train, y_test = dataset.trainSets(x_data, y_data)
    try:
        scaled_x_train, scaled_x_test = svm.dataPreProcess(x_train, x_test)
        print("dataPreProcess Function Executed")
    except:
        print("Failed to execute dataPreProcess()")
    try:
        t0 = time.time()
        svm_clf, y_predict_train, y_predict_test = svm.SVCClassifier(
            scaled_x_train, scaled_x_test, y_train)
        print("seconds wall time:", time.time() - t0)
        print("SVCClassifier Function Executed")
    except:
        print("Failed to execute SVCClassifier()")
    try:
        print("Support Vector Machine Train Accuracy: ",
              svm.SVCTrainAccuracy(y_train, y_predict_train))
    except:
        print("Failed to execute SVCTrainAccuracy()")
    try:
        print("Support Vector Machine Test Accuracy: ",
              svm.SVCTestAccuracy(y_test, y_predict_test))
    except:
        print("Failed to execute SVCTestAccuracy()")
    try:
        print("Plot_ROC_Curve Function Executed",
              svm.ReceiverOperatingCharacteristic(svm_clf, scaled_x_test, y_test))
    except:
        print("Failed to Plot_ROC_Curve")
    try:
        best_params, y_predict_train, y_predict_test = svm.hyperParameterTuning(
            scaled_x_train, scaled_x_test, y_train)
        print("HyperParameterTuning Function Executed")
        print("Best Parameters:", best_params)
        print("Support Vector Machine Train Accuracy: ",
              svm.SVCTrainAccuracy(y_train, y_predict_train))
        print("Support Vector Machine Test Accuracy: ",
              svm.SVCTestAccuracy(y_test, y_predict_test))
    except:
        print("Failed to execute hyperParameterTuning()")


def kNearestNeighborsTest(Data, kNearestNeighbors):
    dataset = Data()
    knn = kNearestNeighbors()
    data = 'data/pima-indians-diabetes.csv'
    x_data, y_data = dataset.dataAllocation(data)
    x_train, x_test, y_train, y_test = dataset.trainSets(x_data, y_data)
    try:
        scaled_x_train, scaled_x_test = knn.dataPreProcess(x_train, x_test)
        print("dataPreProcess Function Executed")
    except:
        print("Failed to execute dataPreProcess()")
    try:
        t0 = time.time()
        knn_clf, y_predict_train, y_predict_test = knn.kNNClassifier(
            scaled_x_train, scaled_x_test, y_train)
        print("seconds wall time:", time.time() - t0)
        print("kNNClassifier Function Executed")
    except:
        print("Failed to execute kNNClassifier()")
    try:
        print("k-Nearest Neighbors Train Accuracy: ",
              knn.KNNTrainAccuracy(y_train, y_predict_train))
    except:
        print("Failed to execute KNNTrainAccuracy()")
    try:
        print("k-Nearest Neighbors Test Accuracy: ",
              knn.KNNTestAccuracy(y_test, y_predict_test))
    except:
        print("Failed to execute KNNTestAccuracy()")
    try:
        print("Plot_ROC_Curve Function Executed",
              knn.ReceiverOperatingCharacteristic(knn_clf, scaled_x_test, y_test))
    except:
        print("Failed to Plot_ROC_Curve")
    try:
        best_params, y_predict_train, y_predict_test = knn.hyperParameterTuning(
            scaled_x_train, scaled_x_test, y_train)
        print("HyperParameterTuning Function Executed")
        print("Best Parameters:", best_params)
        print("k-Nearest Neighbors Train Accuracy: ",
              knn.KNNTrainAccuracy(y_train, y_predict_train))
        print("k-Nearest Neighbors Test Accuracy: ",
              knn.KNNTestAccuracy(y_test, y_predict_test))
    except:
        print("Failed to execute hyperParameterTuning()")
