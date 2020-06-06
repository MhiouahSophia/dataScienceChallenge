from process_data_DL import fastText1_process, fastText2_process, doc2vec_process, dow_keras
from deep_learning_models import CNN_architecture
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
import numpy as np


def train_dl(X_train, X_test, Y_train, Y_test, output_dir, dl_model_name, fastText1, fastText2, job_number, batch_size,
                         num_epochs, num_filters):
    if fastText1:
        X_train_word_seq, X_test_word_seq, max_seq_len, nb_words, embedding_matrix, embed_dim =\
            fastText1_process(X_train, X_test, output_dir,
                      embed_dim=300)
    if fastText2:
        X_train_word_seq, X_test_word_seq, max_seq_len, nb_words, embedding_matrix, embed_dim =\
            fastText2_process(X_train, X_test, output_dir)

    if dl_model_name == 'CNN':
        print('num_filters', num_filters)
        Y_predict, Y_test = CNN_architecture(X_train_word_seq, X_test_word_seq, Y_train, Y_test, max_seq_len, nb_words,
                                             embedding_matrix, embed_dim, output_dir, job_number,
                                            batch_size, num_epochs,
                                             num_filters, weight_decay=1e-4)
    print(' ******** Training  done ')
    print(Y_predict.shape)
    print(Y_test.shape)
    print(Y_test[0])
    print(Y_predict[0])
    auc = roc_auc_score(np.array(Y_test), np.array(Y_predict), average='macro')
    print('Auc score', auc)
