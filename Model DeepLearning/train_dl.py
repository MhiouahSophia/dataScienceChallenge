from configparser import ConfigParser
from cleaning_data import cleaning_text
from split_data import split_train_test
from processing_data import word_count, tfidf, word2vec_fastText
from machine_learning_models import train_RF, train_LR
from calculate_ml_performance_metrics import multiclass_roc_auc_score, conf_matrix


def main():
    # parser config
    config_file = "../config_ml.ini"
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

    if fatstext_:
        X_train, X_test = word2vec_fastText(X_train), word2vec_fastText(X_test)
        print(' ******** Word2vec done ')


if __name__ == "__main__":
    main()
