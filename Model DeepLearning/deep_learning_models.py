import tensorflow as tf
import numpy as np
import keras
from keras import optimizers
from keras import backend as K
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.utils import plot_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils


###############################################
def CNN_architecture(X_train_word_seq, X_test_word_seq, Y_train, Y_test, max_seq_len, nb_words, embedding_matrix,
                    embed_dim, output_dir, job_number, batch_size, num_epochs, num_filters, weight_decay):

    num_classes = np.max(Y_train) + 1

    #CNN architecture
    print("training CNN ...")
    model = Sequential()
    model.add(Embedding(nb_words, embed_dim,
              weights=[embedding_matrix], input_length=max_seq_len, trainable=False))
    model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Dense(num_classes, activation='sigmoid'))  #multi-label (k-hot encoding)

    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()
    #define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
    callbacks_list = [early_stopping]

    # to categorical TO DO

    Y_train = utils.to_categorical(Y_train, num_classes)
    Y_test = utils.to_categorical(Y_test, num_classes)
    #model training
    hist = model.fit(X_train_word_seq, Y_train, batch_size=batch_size, epochs=num_epochs, callbacks=callbacks_list,
                     validation_split=0.1, shuffle=True, verbose=2)

    Y_predict = model.predict(X_test_word_seq)

    np.save(str(output_dir) + 'Y_predict.npy', Y_predict)
    np.save(str(output_dir) + 'Y_test.npy', Y_test)
    model.save(str(output_dir) + 'model' + str(job_number))

    return Y_predict,  Y_test


#
# ###############################################
# # MLP architecture
# num_classes = np.max(y_train) + 1
# y_train = utils.to_categorical(y_train, num_classes)
# y_test = utils.to_categorical(y_test, num_classes)
#
# batch_size = 32
# epochs = 2
#
# # Build the model
# model = Sequential()
# model.add(Dense(512, input_shape=(max_words,)))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes))
# model.add(Activation('softmax'))
#
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
#
# history = model.fit(x_train, y_train,
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=1,
#                     validation_split=0.1)


###############################################
# LSTM architecture
# deep_inputs = Input(shape=(maxlen,))
# embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=False)(deep_inputs)
# LSTM_Layer_1 = LSTM(128)(embedding_layer)
# dense_layer_1 = Dense(6, activation='sigmoid')(LSTM_Layer_1)
# model = Model(inputs=deep_inputs, outputs=dense_layer_1)
#
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])