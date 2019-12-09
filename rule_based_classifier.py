import util as util
import Orange.data
import Orange
import Orange.classification
from sklearn.tree import _tree
import model_evaluation as meval
from sklearn.model_selection import train_test_split
import wittgenstein as lw
  
def rule_based_classifier(training_data):
    print('Generating the data model for a rule based classifier . . .\n')
    X = util.drop_target_variable(training_data)
    y = util.retrieve_target_variable(training_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=1)
    rule_based_classifier=lw.RIPPER()
    rule_based_classifier.fit(X_train, y_train)
    print(rule_based_classifier.ruleset_.out_pretty())
    print('The data model for rule based classifier has been generated successfully!\n')
    util.save_data_model(rule_based_classifier,'rule_based_classifier')
    return;

def rule_based_classifier_prediction(rule_based_data_model,test_data):
    print('Predicting the class label using the rule based classifier model . . .')
    X = util.drop_target_variable(test_data)
    y = util.retrieve_target_variable(test_data)
    meval.plot_ROC_curve(rule_based_data_model, X, y,'Rule Based Classifier')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9999, random_state=1)
    Y_predict = rule_based_data_model.predict(X_test)
    util.display_all_metrics(y_test, Y_predict)
    util.save_data_model(rule_based_data_model,'predicted_rule_based_classifier')
    return;