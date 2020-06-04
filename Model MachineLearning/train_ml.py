from process_data_ML import word_count, tfidf
from ml_models import train_RF, train_LR
from ml_perf_metrics import multiclass_roc_auc_score, conf_matrix


def train_ml(X_train,X_test, Y_train,Y_test,ml_model_name, tfidf_, wordcount_, n_estimators_rf, max_features_rf,
             random_state, max_iter_LR, output_dir, job_number):
    if tfidf_:
        X_train, X_test = tfidf(X_train), tfidf(X_test)
        print(' ******** TFIDF process done ')

        if ml_model_name == 'RF':
            Y_predict = train_RF(X_train, Y_train, X_test, n_estimators_rf, max_features_rf, random_state)
        if ml_model_name == 'LR':
            Y_predict = train_LR(X_train, Y_train, X_test, random_state, max_iter_LR)

        print(' ******** Training  done ')
        auc = multiclass_roc_auc_score(Y_test, Y_predict, average="macro")
        print('Auc score', auc)
        conf_matrix(Y_test, Y_predict, output_dir, str(tfidf) + str(ml_model_name) + str(job_number))

    if wordcount_:
        X_train, X_test = word_count(X_test), word_count(X_test)
        print(' ******** WordCount process done ')
        if ml_model_name == 'RF':
            Y_predict = train_RF(X_train, Y_train, X_test, n_estimators_rf, max_features_rf, random_state)
        if ml_model_name == 'LR':
            Y_predict = train_LR(X_train, Y_train, X_test, random_state, max_iter_LR)

        print(' ******** Training  done ')
        auc = multiclass_roc_auc_score(Y_test, Y_predict, average="macro")
        print('Auc score', auc)
        conf_matrix(Y_test, Y_predict, output_dir, str(tfidf) + str(ml_model_name) + str(job_number))


