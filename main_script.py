import load_dataset as ld
import data_preprocessing as dp
import data_visualization as dv
import feature_selection as fs
import linear_models as ln
import decision_tree_classifier as tree
import nearest_neighbours_classifier as nnc
import rule_based_classifier as rule
import ensemble_classifiers as enc
import util as util


#Reading the .csv files which contains training and test data
print('\n#################### LOADING THE DATA SETS ####################\n')
training_data=ld.load_training_dataset()
test_data=ld.load_test_dataset()


print('\n#################### DATA PREPROCESSING #######################')
print('\n#################### A) DATA TRANSFORMATION ###################\n')
dp.check_for_missing_values(training_data)
print('\n#################### B) DATA REDUCTION ########################\n')
test_data=dp.duplicate_feature_elimination(test_data)


print('\n#################### DATA VISUALIZATION #######################\n')
#Visualizing the age distribution
dv.visualize_any_feature_as_distribution_plot(training_data,'age')
#Visualizing the loan amount distribution
dv.visualize_any_feature_as_distribution_plot(training_data,'loan_size')
#Checking for class imbalance based on gender
dv.convert_categorical_data_to_numerical_data(training_data)


#Eliminating the features 'ZIP' and 'occupation' after feature engineering
print('\n#################### FEATURE ENGINEERING ####################\n')
print('Eliminating the feature \'ZIP\' after feature engineering . . .')
training_data=fs.feature_elimination_zip(training_data)
print('\nEliminating the feature \'occupation\' after feature engineering . . .')
training_data=fs.feature_elimination_occupation(training_data)


print('\n#################### MODEL CONSTRUCTION ####################\n')
print('\n#################### A) LINEAR SUPPORT VECTOR CLASSIFIER ####################\n')
ln.linear_support_vector_classifier(training_data)
print('\n#################### B) DECISION TREE CLASSIFIER ####################\n')
tree.decision_tree_classifier(training_data)
print('\n#################### C) NEAREST NEIGHBOURS (DISTANCE-BASED) CLASSIFIER ####################\n')
knn=nnc.nearest_neighbours_classifier(training_data)
print('\n#################### D) RULE BASED CLASSIFIER ####################\n')
#rule.rule_based_classifier(training_data)
print('\n#################### E) ENSEMBLE MODELS ####################\n')
print('\n#################### i) BAGGING #############################\n')
enc.bagging_method_classifier(training_data, knn)
print('\n#################### ii) RANDOM FORESTS ##########################\n')
enc.random_forest_classifier(training_data)
print('\n#################### iii) ADA BOOST ##########################\n')
enc.adaboost_classifier(training_data)
print('\n#################### iv) STOCHASTIC GRADIENT BOOSTING ##########################\n')
enc.stochastic_gradient_boosting_classifier(training_data)


print('\n#################### MODEL EVALUATION AND TESTING ####################\n')
print('\n#################### A) LINEAR SUPPORT VECTOR CLASSIFIER ####################\n')
linear_support_vector_data_model=util.load_data_model('linear_support_vector_classifier')
ln.linear_support_vector_classifier_prediction(linear_support_vector_data_model, test_data)
print('\n#################### B) DECISION TREE CLASSIFIER ####################\n')
decision_tree_data_model=util.load_data_model('decision_tree_classifier')
tree.decision_tree_classifier_prediction(decision_tree_data_model,test_data)
print('\n#################### C) NEAREST NEIGHBOURS (DISTANCE-BASED) CLASSIFIER ####################\n')
nearest_neighbours_data_model=util.load_data_model('nearest_neighbours_classifier')
nnc.nearest_neighbours_classifier_prediction(nearest_neighbours_data_model,test_data)
print('\n#################### D) RULE BASED CLASSIFIER ####################\n')
rule_based_data_model=util.load_data_model('rule_based_classifier')
#rule.rule_based_classifier_prediction(rule_based_data_model, test_data)
print('\n#################### E) ENSEMBLE MODELS ####################\n')
print('\n#################### i) BAGGING #############################\n')
bagging_method_data_model=util.load_data_model('bagging_method_classifier')
enc.bagging_method_classifier_prediction(bagging_method_data_model, test_data)
print('\n#################### ii) RANDOM FORESTS ##########################\n')
random_forest_data_model=util.load_data_model('random_forest_classifier')
enc.random_forest_classifier_prediction(random_forest_data_model, test_data)
print('\n#################### iii) ADA BOOST ##########################\n')
adaboost_data_model=util.load_data_model('adaboost_classifier')
enc.adaboost_classifier_prediction(adaboost_data_model, test_data)
print('\n#################### iv) STOCHASTIC GRADIENT BOOSTING ##########################\n')
stochastic_gradient_boosting_data_model=util.load_data_model('stochastic_gradient_boosting_classifier')
enc.stochastic_gradient_boosting_classifier_prediction(stochastic_gradient_boosting_data_model, test_data)