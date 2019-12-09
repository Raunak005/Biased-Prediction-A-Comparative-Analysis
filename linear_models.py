import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import util as util
import model_evaluation as meval
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification

def linear_support_vector_classifier(training_data):
    print('Generating the data model for a linear support vector classifier . . .\n')
    X, y = make_classification(n_features=11, random_state=0)
    linear_support_vector_classifier = LinearSVC(random_state=0, tol=1e-5)
    linear_support_vector_classifier.fit(X, y)
    print('The intercept of linear support vector classifier is : ',linear_support_vector_classifier.intercept_)
    print('\nThe slope of linear support vector classifier is : ',linear_support_vector_classifier.coef_)
    print('\nThe data model for linear support vector classifier has been generated successfully!\n')
    util.save_data_model(linear_support_vector_classifier,'linear_support_vector_classifier')
#     meval.plot_ROC_curve(linear_support_vector_classifier, X, y,'Linear Support Vector Classifier')
    return;

def linear_support_vector_classifier_prediction(linear_support_vector_data_model,test_data):
    X, y = make_classification(n_features=11, random_state=0)
    print('Predicting the class label using the linear support vector classifier model . . .',linear_support_vector_data_model.predict([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
#     meval.plot_ROC_curve(linear_support_vector_data_model, X, y,'Linear Support Vector Classifier')
    util.save_data_model(linear_support_vector_data_model,'predicted_linear_support_vector_classifier')
    return;