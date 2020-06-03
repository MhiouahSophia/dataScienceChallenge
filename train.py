from configparser import ConfigParser
from cleaning_data import cleaning_text
from split_data import split_train_test
from train_ml import train_ml
import os


def main():
    # parser config
    config_file = "../config.ini"
    cp = ConfigParser()
    cp.read(config_file)

    # default config
    job_number = cp["DATA"].getint("job_number")
    model = cp["DATA"].get("model")
    output_dir = cp["DATA"].get("output_dir") + str(model) + '/' + str(job_number) + '/'
    # create folder for writing the result

    if os.path.exists(output_dir):
        print('This number job as already been launcned')
        pass

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

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
    print('outputdir ', output_dir)
    with open(str(output_dir) + 'config_jobnumber' + str(job_number) + '.ini', 'w') as configfile:
        cp.write(configfile)

    df_feedback_clean = cleaning_text(data_path, remove_nan)
    print(df_feedback_clean.head())
    X_train, X_test, Y_train, Y_test = split_train_test(df_feedback_clean)
    print(X_train.head())
    print(' ******** Loaded, cleaned and split the data')

    if model == 'ML':
        train_ml(X_train, X_test, Y_train, Y_test, ml_model_name, tfidf_, wordcount_, n_estimators_rf, max_features_rf,
                 random_state, max_iter_LR, output_dir, job_number)


if __name__ == "__main__":
    main()

