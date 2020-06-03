from configparser import ConfigParser
from cleaning_data import cleaning_text
from split_data import split_train_test
from processing_data import word_count, tfidf, word2vec_fastText
from machine_learning_models import train_RF, train_LR
from calculate_performance_metrics import multiclass_roc_auc_score


def main():
    # parser config
    config_file = "..//config.ini"
    cp = ConfigParser()
    cp.read(config_file)

    # default config
    output_dir = cp["DATA"].get("output_dir")
    data_path = cp["DATA"].get("data_path")
    remove_nan = cp["DATA"].getboolean("remove_nan")
    random_state = cp["DATA"].getint("random_state")

    tfidf_ = cp["PROCESSING"].getboolean("tfidf_")
    wordcount_ = cp["PROCESSING"].getboolean("wordcount_")
    fatstext_ = cp["PROCESSING"].getboolean("fatstext_")

    ml_model_name = cp["ML"].get("ml_model_name")

    n_estimators_rf = cp["ML_RF"].getint("n_estimators_rf")
    max_features_rf = cp["ML_RF"].getint("max_features_rf")

    max_iter_LR = cp["ML_LR"].getint("max_iter_LR")

    df_feedback_clean = cleaning_text(data_path, remove_nan)
    print(df_feedback_clean.head())
    X_train, X_test, Y_train, Y_test = split_train_test(df_feedback_clean)
    print(X_train.head())
    print(' ******** Loaded, cleaned and split the data')

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

    if fatstext_:
        X_train, X_test = word2vec_fastText(X_train), word2vec_fastText(X_test)
        print(' ******** Word2vec done ')




if __name__ == "__main__":
    main()
