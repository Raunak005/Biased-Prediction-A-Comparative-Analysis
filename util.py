from sklearn.metrics import classification_report, confusion_matrix
from sklearn import model_selection
from sklearn import metrics
import pickle

#Dropping the class label column from the training data
def drop_target_variable(training_data):
    X = training_data.drop('default', axis=1)
    return X;

#Retrieving the class label column
def retrieve_target_variable(training_data):
    y = training_data['default']
    return y;

#Displaying the metrics
def display_all_metrics(y_test, y_pred):
    print('\nThe confusion matrix is :')
    print(confusion_matrix(y_test, y_pred))
    print('\nThe classification report is :')
    print(classification_report(y_test, y_pred))
    print('\nThe accuracy of the model is : %.3f' % ((metrics.accuracy_score(y_test, y_pred))*100),'%','\n')
    return;

#Saving the data model
def save_data_model(classifier,classifier_name):
    model_file_name='./models/'+classifier_name+'_model.sav'
    pickle.dump(classifier,open(model_file_name,'wb'))
    print('The data model for',classifier_name,'has been saved successfully!')
    return;

#Loading the saved data model
def load_data_model(classifier_name):
    model_file_name='./models/'+classifier_name+'_model.sav'
    loaded_data_model=pickle.load(open(model_file_name,'rb'))
    print('The saved data model for',classifier_name,'has been loaded successfully!')
    return loaded_data_model;