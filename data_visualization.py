import matplotlib.pyplot as pyplot
import seaborn as seabrn

def visualize_any_feature_as_distribution_plot(training_data,feature_name):
    print('Visualizing the feature - '+ feature_name +' in the training data set . . .')
    pyplot.figure(figsize=(10,5))
    pyplot.tight_layout()
    seabrn.distplot(training_data[feature_name])
    return;

#Converting 'Default' to 1 and 'Non-Default' to 0
def convert_categorical_data_to_numerical_data(training_data):
    print('Visualizing the training data set . . .')
    training_data['default'] = training_data['default'].map({True: 1, False: 0})
    training_data.plot(x='sex', y='default', style='o')
    pyplot.title('Gender v/s Defaulter')  
    pyplot.xlabel('Gender')  
    pyplot.ylabel('Defaulter')  
    pyplot.show()
    return;