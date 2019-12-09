import pandas as pd
import feature_selection as fs

def load_training_dataset():
    print('Loading the training data set . . .')
    training_data = pd.read_csv('./datasets/train.csv')
    print('The training data set (\'train.csv\') is loaded successfully!')
    return training_data;

def load_test_dataset():
    print('Loading the test data set . . .')
    test_data = pd.read_csv('./datasets/test.csv')
    test_data=fs.feature_elimination_zip(test_data)
    test_data=fs.feature_elimination_occupation(test_data)
    print('The test data set (\'test.csv\') is loaded successfully!')
    return test_data;