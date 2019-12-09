from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import decision_tree_classifier as tree
import model_evaluation as meval
import util as util

def bagging_method_classifier(training_data,nearest_neighbours_classifier):
    print('Generating the data model for a bagging method classifier . . .\n')
    bagging_method_classifier=BaggingClassifier(base_estimator=nearest_neighbours_classifier,n_estimators=100,random_state=7,max_samples=0.5, max_features=0.5)
    X = util.drop_target_variable(training_data)
    y = util.retrieve_target_variable(training_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y, test_size=0.1, random_state=1)
    bagging_method_classifier=bagging_method_classifier.fit(X_train, y_train)
    print('The data model for bagging method classifier has been generated successfully!\n')
    util.save_data_model(bagging_method_classifier,'bagging_method_classifier')
    return;

def bagging_method_classifier_prediction(bagging_method_data_model,test_data):
    print('Predicting the class label using the bagging method classifier model . . .')
    X = util.drop_target_variable(test_data)
    y = util.retrieve_target_variable(test_data)
    meval.plot_ROC_curve(bagging_method_data_model, X, y,'Bagging Method Classifier')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9999, random_state=1)
    Y_predict = bagging_method_data_model.predict(X_test)
    util.display_all_metrics(y_test, Y_predict)
    util.save_data_model(bagging_method_data_model,'predicted_bagging_method_classifier')
    return;

def random_forest_classifier(training_data):
    print('Generating the data model for random forest classifier . . .\n')
    random_forest_classifier=RandomForestClassifier(n_estimators=100, random_state=7)
    X = util.drop_target_variable(training_data)
    y = util.retrieve_target_variable(training_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y, test_size=0.1, random_state=1)
    random_forest_classifier=random_forest_classifier.fit(X_train, y_train)
    print('The data model for random forest classifier has been generated successfully!\n')
    single_decision_tree = random_forest_classifier.estimators_[5]
    tree.visualizing_decision_tree(single_decision_tree, X, 'visualizing_random_forest')
    util.save_data_model(random_forest_classifier,'random_forest_classifier')
    return;

def random_forest_classifier_prediction(random_forest_data_model,test_data):
    print('Predicting the class label using the random forest classifier model . . .')
    X = util.drop_target_variable(test_data)
    y = util.retrieve_target_variable(test_data)
    meval.plot_ROC_curve(random_forest_data_model, X, y,'Random Forest Classifier')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9999, random_state=1)
    Y_predict = random_forest_data_model.predict(X_test)
    util.display_all_metrics(y_test, Y_predict)
    single_predicted_decision_tree = random_forest_data_model.estimators_[5]
    tree.visualizing_decision_tree(single_predicted_decision_tree, X, 'visualizing_predicted_random_forest')
    util.save_data_model(random_forest_data_model,'predicted_random_forest_classifier')
    return;

def adaboost_classifier(training_data):
    print('Generating the data model for adaboost classifier . . .\n')
    adaboost_classifier=AdaBoostClassifier(n_estimators=100, random_state=7)
    X = util.drop_target_variable(training_data)
    y = util.retrieve_target_variable(training_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y, test_size=0.1, random_state=1)
    adaboost_classifier=adaboost_classifier.fit(X_train, y_train)
    print('The data model for adaboost classifier has been generated successfully!\n')
    util.save_data_model(adaboost_classifier,'adaboost_classifier')
    return;

def adaboost_classifier_prediction(adaboost_data_model,test_data):
    print('Predicting the class label using the adaboost classifier model . . .')
    X = util.drop_target_variable(test_data)
    y = util.retrieve_target_variable(test_data)
    meval.plot_ROC_curve(adaboost_data_model, X, y,'Ada Boost Classifier')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9999, random_state=1)
    Y_predict = adaboost_data_model.predict(X_test)
    util.display_all_metrics(y_test, Y_predict)
    util.save_data_model(adaboost_data_model,'predicted_adaboost_classifier')
    return;

def stochastic_gradient_boosting_classifier(training_data):
    print('Generating the data model for stochastic gradient boosting classifier . . .\n')
    stochastic_gradient_boosting_classifier=GradientBoostingClassifier(n_estimators=100, random_state=7)
    X = util.drop_target_variable(training_data)
    y = util.retrieve_target_variable(training_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y, test_size=0.1, random_state=1)
    stochastic_gradient_boosting_classifier=stochastic_gradient_boosting_classifier.fit(X_train, y_train)
    print('The data model for stochastic gradient boosting classifier has been generated successfully!\n')
    util.save_data_model(stochastic_gradient_boosting_classifier,'stochastic_gradient_boosting_classifier')
    return;

def stochastic_gradient_boosting_classifier_prediction(stochastic_gradient_boosting_data_model,test_data):
    print('Predicting the class label using the stochastic gradient boosting classifier model . . .')
    X = util.drop_target_variable(test_data)
    y = util.retrieve_target_variable(test_data)
    meval.plot_ROC_curve(stochastic_gradient_boosting_data_model, X, y,'Stochastic Gradient Boosting Classifier')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9999, random_state=1)
    Y_predict = stochastic_gradient_boosting_data_model.predict(X_test)
    util.display_all_metrics(y_test, Y_predict)
    util.save_data_model(stochastic_gradient_boosting_data_model,'predicted_stochastic_gradient_boosting_classifier')
    return;