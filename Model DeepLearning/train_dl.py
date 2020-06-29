from process_data_DL import fastText_process, doc2vec_process, dow_keras
from deep_learning_models import CNN_architecture, CNNRNN_architecture, DPCNN_architecture
from tensorflow import keras
from keras import utils

def train_dl(X_train, X_test, Y_train, output_dir, dl_model_name, fastText, job_number, batch_size,
             num_epochs, num_filters, num_classes, best_model):
    if fastText:
        X_train_word_seq, X_test_word_seq, max_seq_len, nb_words, embedding_matrix, embed_dim = \
            fastText_process(X_train, X_test, output_dir)
    if best_model == True:
        Y_train = utils.to_categorical(Y_train, num_classes)
        reconstructed_model = keras.models.load_model("best_model.h5")
        reconstructed_model.fit(X_train_word_seq, Y_train, epochs=1)
        Y_predict = reconstructed_model.predict(X_test_word_seq)

    elif dl_model_name == 'CNN':
        Y_predict = CNN_architecture(X_train_word_seq, X_test_word_seq, Y_train, max_seq_len, nb_words,
                                     embedding_matrix,
                                     embed_dim, output_dir, job_number, batch_size, num_epochs, num_filters,
                                     weight_decay=1e-4)

    elif dl_model_name == 'CNNRNN':
        Y_predict = CNNRNN_architecture(X_train_word_seq, X_test_word_seq, Y_train, max_seq_len, nb_words,
                                        embedding_matrix,
                                        embed_dim, output_dir, job_number, batch_size, num_epochs, num_filters,
                                        weight_decay=1e-4)

    elif dl_model_name == 'DPCNN':
        Y_predict = DPCNN_architecture(X_train_word_seq, X_test_word_seq, Y_train, max_seq_len, nb_words,
                                       embedding_matrix,
                                       embed_dim, output_dir, job_number, batch_size, num_epochs, num_filters,
                                       weight_decay=1e-4)

    print(' ******** Training  done ')
    print(Y_predict.shape)
    return Y_predict


def fasttext(X_train, X_test, output_dir):
    X_train_word_seq, X_test_word_seq, max_seq_len, nb_words, embedding_matrix, embed_dim, df_wnf = \
        fastText_process(X_train, X_test, output_dir)
    return df_wnf

