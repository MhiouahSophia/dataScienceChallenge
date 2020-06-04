from configparser import ConfigParser
from cleaning_data import cleaning_text
from split_data import split_train_test
from train_ml import train_ml
import os
import random
from process_data_DL import fastText1_process, fastText2_process, doc2vec_process, dow_keras
from deep_learning_models import CNN_architecture
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
import numpy as np


def main():
    # parser config
    config_file = "./config.ini"
    cp = ConfigParser()
    cp.read(config_file)

    # default config
    job_number = random.randint(0, 100000000)
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

    fastText1 = cp["PROCESSING_DL"].getboolean("fastText1")
    fastText2 = cp["PROCESSING_DL"].getboolean("fastText2")
    dl_model_name = cp["DL"].get("dl_model_name")
    batch_size = cp["PROCESSING_DL"].getint("batch_size")
    num_epochs = cp["PROCESSING_DL"].getint("num_epochs")
    num_filters = cp["PROCESSING_DL"].getint("num_filters")
    weight_decay = cp["PROCESSING_DL"].getint("weight_decay")

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

    if model == 'DL':
        if fastText1:
            X_train_word_seq, X_test_word_seq, max_seq_len, nb_words, embedding_matrix, embed_dim = \
                fastText1_process(X_train, X_test, MAX_NB_WORDS=100000,
                                  embed_dim=300)

        if dl_model_name == 'CNN':
            Y_predict, Y_test = CNN_architecture(X_train_word_seq, X_test_word_seq, Y_train, Y_test,
                                                 max_seq_len, nb_words, embedding_matrix,
                    embed_dim, output_dir, job_number, batch_size, num_epochs, num_filters, weight_decay)
            
        print(' ******** Training  done ')
        print(Y_predict.shape)
        print(Y_test.shape)
        print(Y_test[0])
        print(Y_predict[0])
        auc = roc_auc_score(np.array(Y_test), np.array(Y_predict), average='macro')
        print('Auc score', auc)


if __name__ == "__main__":
    main()

