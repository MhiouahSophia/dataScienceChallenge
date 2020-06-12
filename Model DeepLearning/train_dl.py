from process_data_DL import fastText2_process, doc2vec_process, dow_keras
from deep_learning_models import CNN_architecture
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
import numpy as np
from keras import utils


def train_dl(X_train, X_test, Y_train, output_dir, dl_model_name, fastText1, fastText2, job_number, batch_size,
                         num_epochs, num_filters, num_classes):
    if fastText2:
        X_train_word_seq, X_test_word_seq, max_seq_len, nb_words, embedding_matrix, embed_dim =\
            fastText2_process(X_train, X_test, output_dir)

    if dl_model_name == 'CNN':
        print('num_filters', num_filters)
        Y_predict = CNN_architecture(X_train_word_seq, X_test_word_seq, Y_train, max_seq_len, nb_words,
                                             embedding_matrix, embed_dim, output_dir, job_number,
                                            batch_size, num_epochs,
                                             num_filters, weight_decay=1e-4)


    print(' ******** Training  done ')
    print(Y_predict.shape)

    return Y_predict
