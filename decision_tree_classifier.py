from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import util as util
import model_evaluation as meval
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

def decision_tree_classifier(training_data):
    print('Generating the data model for a decision tree classifier . . .\n')
    X = util.drop_target_variable(training_data)
    y = util.retrieve_target_variable(training_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
    decision_tree_classifier = DecisionTreeClassifier()
    decision_tree_classifier=decision_tree_classifier.fit(X_train, y_train)
    print('The data model for decision tree classifier has been generated successfully!\n')
    visualizing_decision_tree(decision_tree_classifier,X,'visualizing_decision_tree')
    util.save_data_model(decision_tree_classifier,'decision_tree_classifier')
    return;

def visualizing_decision_tree(decision_tree_classifier,X,image_file_name):
    dot_data = StringIO()
    export_graphviz(decision_tree_classifier, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = X.columns,class_names=['Non-Defaulter','Defaulter'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    image_file_name='./output/'+image_file_name+'.png' 
    graph.write_png(image_file_name)
    Image(graph.create_png())
    print('The decision tree has been saved successfully as an image in the \'output\' directory\n')
    return;

def decision_tree_classifier_prediction(decision_tree_data_model,test_data):
    print('Predicting the class label using the decision tree classifier model . . .')
    X = util.drop_target_variable(test_data)
    y = util.retrieve_target_variable(test_data)
    meval.plot_ROC_curve(decision_tree_data_model, X, y,'Decision Tree Classifier')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9999, random_state=1)
    Y_predict = decision_tree_data_model.predict(X_test)
    util.display_all_metrics(y_test, Y_predict)
    visualizing_decision_tree(decision_tree_data_model,X,'visualizing_predicted_decision_tree')
    util.save_data_model(decision_tree_data_model,'predicted_decision_tree_classifier')
    return;