#Retrieving default independent variable(s)
def retrieve_default_feature_variables():
    default_features = ['minority', 'sex', 'ZIP','rent','education','age','income','loan_size','payment_timing','year','job_stability','occupation']
    print('The default feature (independent) variable(s) are:',default_features)
    return default_features;

#Retrieving default dependent variable(s)
def retrieve_default_target_variables():
    default_target = ['default']
    print('The default dependent variable(s) are:',default_target)
    return default_target;

#Eliminating the feature 'ZIP' after feature engineering
def feature_elimination_zip(training_data):
    training_data=training_data.drop('ZIP', axis=1)
    return training_data;

#Eliminating the feature 'occupation' after feature engineering
def feature_elimination_occupation(training_data):
    training_data=training_data.drop('occupation', axis=1)
    return training_data;