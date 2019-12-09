from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc

def plot_ROC_curve(data_model,X,y,model_label):
    print('\n*************** EVALUATION CRITERIA BEGINS ***************')
    print('\nGenerating the ROC curve for model evaluation . . .')
    trainX, testX, train_y, test_y = train_test_split(X, y, test_size=0.5, random_state=2)
    no_skill_prediction_probability = [0 for _ in range(len(test_y))]
    data_model.fit(trainX, train_y)
    data_model_prediction_probability = data_model.predict_proba(testX)
    data_model_prediction_probability = data_model_prediction_probability[:, 1]
    data_model_fpr, data_model_tpr, _ = roc_curve(test_y, data_model_prediction_probability)
    no_skill_fpr, no_skill_tpr, _ = roc_curve(test_y, no_skill_prediction_probability)
    pyplot.plot(no_skill_fpr, no_skill_tpr, linestyle='--', label='No Skill')
    pyplot.plot(data_model_fpr, data_model_tpr, marker='.', label=model_label)
    pyplot.xlabel('False Positive Rate ->')
    pyplot.ylabel('True Positive Rate ->')
    pyplot.legend()
    pyplot.show()
    calculate_AUC_value(test_y, data_model_prediction_probability, no_skill_prediction_probability, model_label)
    if model_label!= 'Rule Based Classifier':
        plot_precision_recall_curve(data_model, testX, test_y, data_model_prediction_probability, model_label)
    print('\n*************** EVALUATION CRITERIA ENDS *****************')
    return;

def calculate_AUC_value(test_y,data_model_prediction_probability,no_skill_prediction_probability,model_label):
    data_model_auc = roc_auc_score(test_y, data_model_prediction_probability)
    no_skill_auc = roc_auc_score(test_y, no_skill_prediction_probability)
    print('\nThe area under the ROC curve (AUC) of the '+model_label+' is = %.3f' % (data_model_auc))
    print('\nThe area under the ROC curve (AUC) of no skill learning is = %.3f' % (no_skill_auc))
    return;

def plot_precision_recall_curve(data_model,testX,test_y,data_model_prediction_probability,model_label):
    print('\nGenerating the Precision - Recall curve for model evaluation . . .')
    data_model_prediction = data_model.predict(testX)
    data_model_precision, data_model_recall, _ = precision_recall_curve(test_y, data_model_prediction_probability)
    data_model_f1, data_model_auc = f1_score(test_y, data_model_prediction), auc(data_model_recall, data_model_precision)
    calculate_AUC_value_and_F1_score(data_model_auc, data_model_f1, model_label)
    no_skill_prediction = len(data_model_prediction[data_model_prediction==1]) / len(data_model_prediction)
    pyplot.plot([0, 1], [no_skill_prediction, no_skill_prediction], linestyle='--', label='No Skill')
    pyplot.plot(data_model_recall, data_model_precision, marker='.', label=model_label)
    pyplot.xlabel('Recall ->')
    pyplot.ylabel('Precision ->')
    pyplot.legend()
    pyplot.show()
    return;

def calculate_AUC_value_and_F1_score(data_model_auc,data_model_f1,model_label):
    print('\nThe area under the Precision - Recall curve (AUC) of the '+model_label+' is = %.3f' % (data_model_auc))
    print('\nThe F1 score of the '+ model_label +' is = %.3f' % (data_model_f1))
    return;