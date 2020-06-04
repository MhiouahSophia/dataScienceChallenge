# Metrics in order to evaluate the model performance
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer


def conf_matrix(Y_test, Y_predict, output_path, name_fig):
    df_cm = pd.DataFrame(confusion_matrix(Y_test, Y_predict))
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True)
    plt.title('Score AUC ' + str(round(multiclass_roc_auc_score(Y_test, Y_predict, average="macro"), 2)))
    plt.savefig(output_path + str(name_fig))
    return df_cm


def accuracy(Y_test, Y_predict):
    print("Accuracy score Quadratic classifier fastext: " + str(
        round(accuracy_score(Y_test, Y_predict) * 100)) + "%")
    return round(accuracy_score(Y_test, Y_predict) * 100)


def auc(Y_test, Y_predict):
    auc = roc_auc_score(Y_test, Y_predict)
    return auc


def multiclass_roc_auc_score(Y_test, Y_predict, average="macro"):
    lb = LabelBinarizer()
    lb.fit(Y_test)
    y_test = lb.transform(Y_test)
    y_pred = lb.transform(Y_predict)
    return roc_auc_score(y_test, y_pred, average=average)


def f1score(Y_test, Y_predict):
    f1score = f1_score(Y_test, Y_predict)
    return f1score
