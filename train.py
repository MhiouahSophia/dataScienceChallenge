from datetime import datetime
from configparser import ConfigParser
from cleaning_data import cleaning_text, class_weight
from split_data import split_train_test
from train_ml import train_ml
import os
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
import numpy as np
from train_dl import train_dl
from keras import utils

# for experimentation purpose/ evaluation of the model on 20 % of the training data


def main():
    # parser config
    config_file = "./config.ini"
    cp = ConfigParser()
    cp.read(config_file)

    # default config
    # datetime object containing current date and time
    now = datetime.now()
    print(now)
    print(now.strftime("%d/%m/%Y %H:%M:%S"))
    job_number = now.strftime("%d-%m-%Y_%H:%M:%S")
    print('job_number', job_number)

    model = cp["DEFAULT"].get("model")
    output_dir = cp["DEFAULT"].get("output_dir") + str(model) + '/' + str(job_number) + '/'
    # create folder for writing the result

    if os.path.exists(output_dir):
        print('This number job as already been launched')
        pass

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    data_path = cp["DEFAULT"].get("data_path")
    remove_nan = cp["DEFAULT"].getboolean("remove_nan")
    random_state = cp["DEFAULT"].getint("random_state")

    tfidf_ = cp["PROCESSING_ML"].getboolean("tfidf_")
    wordcount_ = cp["PROCESSING_ML"].getboolean("wordcount_")
    ml_model_name = cp["ML"].get("ml_model_name")
    n_estimators_rf = cp["ML_RF"].getint("n_estimators_rf")
    max_features_rf = cp["ML_RF"].getint("max_features_rf")
    max_iter_LR = cp["ML_LR"].getint("max_iter_LR")

    fastText = cp["PROCESSING_DL"].getboolean("fastText")
    dl_model_name = cp["DL"].get("dl_model_name")
    batch_size = cp["CNN_PARA"].getint("batch_size")
    num_epochs = cp["CNN_PARA"].getint("num_epochs")
    num_filters = cp["CNN_PARA"].getint("num_filters")

    print('outputdir ', output_dir)
    with open(str(output_dir) + 'config_jobnumber' + str(job_number) + '.ini', 'w') as configfile:
        cp.write(configfile)

    df_feedback_clean = cleaning_text(data_path, remove_nan)
    print(df_feedback_clean.head())
    class_w = class_weight(df_feedback_clean)
    print(class_w)
    X_train, X_test, Y_train, Y_test = split_train_test(df_feedback_clean, output_dir, job_number)
    print(X_train.head())
    num_classes = np.max(Y_train) + 1
    print(' ******** Loaded, cleaned and split the data')

    if model == 'ML':
        train_ml(X_train, X_test, Y_train, Y_test, ml_model_name, tfidf_, wordcount_, n_estimators_rf, max_features_rf,
                 random_state, max_iter_LR, output_dir, job_number)

    if model == 'DL':
        Y_predict = train_dl(X_train, X_test, Y_train, output_dir, dl_model_name, fastText, job_number,
                             batch_size,
                             num_epochs, num_filters, num_classes)

        Y_test = utils.to_categorical(Y_test, num_classes)
        print('auc', roc_auc_score(Y_test, Y_predict))

        np.save(str(output_dir) + 'Y_test.npy', Y_test)
        print('Y_test', Y_test)


if __name__ == "__main__":
    main()

