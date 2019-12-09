from sklearn.neighbors import (NeighborhoodComponentsAnalysis,KNeighborsClassifier)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import util as util
import model_evaluation as meval

def nearest_neighbours_classifier(training_data):
    print('Generating the data model for a nearest neighbours classifier . . .\n')
    X = util.drop_target_variable(training_data)
    y = util.retrieve_target_variable(training_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y, test_size=0.1, random_state=1)
    nca = NeighborhoodComponentsAnalysis(random_state=42)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn=knn.fit(X_train, y_train)
    nca_pipe = Pipeline([('nca', nca), ('knn', knn)])
    nca_pipe.fit(X_train, y_train)
    print('The data model for nearest neighbours classifier has been generated successfully!\n')
    util.save_data_model(knn,'nearest_neighbours_classifier')
    return knn;

def nearest_neighbours_classifier_prediction(nearest_neighbours_data_model,test_data):
    print('Predicting the class label using the nearest neighbours classifier model . . .')
    X = util.drop_target_variable(test_data)
    y = util.retrieve_target_variable(test_data)
    meval.plot_ROC_curve(nearest_neighbours_data_model, X, y,'Nearest Neighbours Classifier')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9999, random_state=1)
    Y_predict = nearest_neighbours_data_model.predict(X_test)
    util.display_all_metrics(y_test, Y_predict)
    util.save_data_model(nearest_neighbours_data_model,'predicted_nearest_neighbours_classifier')
    return;