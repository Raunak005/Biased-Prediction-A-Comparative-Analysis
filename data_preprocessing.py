def check_for_missing_values(training_data):
    print('Scanning the data set for any missing values . . .')
    print('The break down of the total number of null values in the data set per feature is :')
    print(training_data.isnull().sum())
    return;

def duplicate_feature_elimination(test_data):
    print('Eliminating the duplicate data present in the data')
    test_data=test_data.drop('Unnamed: 0', axis=1)
    return test_data;