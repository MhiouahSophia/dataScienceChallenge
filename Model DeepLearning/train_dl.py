from process_data_DL import fastText2_process, doc2vec_process, dow_keras
from deep_learning_models import CNN_architecture, CNNRNN_architecture, DPCNN_architecture


def train_dl(X_train, X_test, Y_train, output_dir, dl_model_name, fastText1, fastText2, job_number, batch_size,
             num_epochs, num_filters, num_classes):
    if fastText2:
        X_train_word_seq, X_test_word_seq, max_seq_len, nb_words, embedding_matrix, embed_dim = \
            fastText2_process(X_train, X_test, output_dir)
    print('X_test', X_test)
    print('X_train', X_train)

    if dl_model_name == 'CNN':
        print('num_filters', num_filters)
        Y_predict = CNN_architecture(X_train_word_seq, X_test_word_seq, Y_train, max_seq_len, nb_words,
                                     embedding_matrix,
                                     embed_dim, output_dir, job_number, batch_size, num_epochs, num_filters,
                                     weight_decay=1e-4)

    if dl_model_name == 'CNNRNN':
        print('num_filters', num_filters)
        Y_predict = CNNRNN_architecture(X_train_word_seq, X_test_word_seq, Y_train, max_seq_len, nb_words,
                                        embedding_matrix,
                                        embed_dim, output_dir, job_number, batch_size, num_epochs, num_filters,
                                        weight_decay=1e-4)

    if dl_model_name == 'DPCNN':
        print('num_filters', num_filters)
        Y_predict = DPCNN_architecture(X_train_word_seq, X_test_word_seq, Y_train, max_seq_len, nb_words,
                                       embedding_matrix,
                                       embed_dim, output_dir, job_number, batch_size, num_epochs, num_filters,
                                       weight_decay=1e-4)

    print(' ******** Training  done ')
    print(Y_predict.shape)
    return Y_predict


def fasttext(X_train, X_test, output_dir):
    X_train_word_seq, X_test_word_seq, max_seq_len, nb_words, embedding_matrix, embed_dim, df_wnf = \
        fastText2_process(X_train, X_test, output_dir)
    return df_wnf

