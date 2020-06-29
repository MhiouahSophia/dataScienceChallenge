import tensorflow as tf
import numpy as np
from keras import optimizers
from keras import regularizers
from keras import metrics
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Embedding, Conv1D, SpatialDropout1D, MaxPooling1D, GlobalMaxPooling1D, Bidirectional, GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import utils
import sys

###############################################


def CNN_architecture(X_train_word_seq, X_test_word_seq, Y_train, max_seq_len, nb_words, embedding_matrix,
                     embed_dim, output_dir, job_number, batch_size, num_epochs, num_filters, weight_decay):
    num_classes = np.max(Y_train) + 1

    # CNN architecture
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
    model.add(Dense(num_classes, activation='sigmoid'))  # multi-label (k-hot encoding)

    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=[metrics.AUC()])
    # checkpoint
    filepath = output_dir
    checkpoint = ModelCheckpoint(filepath + "best.h5", monitor='val_accuracy', verbose=1, save_best_only=True,
                                 mode='max')
    model.summary()
    # define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=5, verbose=1)
    callbacks_list = [early_stopping, checkpoint]

    # to categorical TO DO

    Y_train = utils.to_categorical(Y_train, num_classes)
    # model training
    hist = model.fit(X_train_word_seq, Y_train, batch_size=batch_size, epochs=num_epochs, callbacks=callbacks_list,
                     validation_split=0.1, shuffle=True, verbose=2)
    print('X_test_word_seq.shape', X_test_word_seq.shape)
    Y_predict = model.predict(X_test_word_seq)
    print('Y_predict', Y_predict)
    np.save(str(output_dir) + 'Y_predict.npy', Y_predict)
    model.save(str(output_dir) + 'model' + str(job_number) + '.h5')

    return Y_predict


###############################################
def CNNRNN_architecture(X_train_word_seq, X_test_word_seq, Y_train, max_seq_len, nb_words, embedding_matrix,
                        embed_dim, output_dir, job_number, batch_size, num_epochs, num_filters, weight_decay):
    num_classes = np.max(Y_train) + 1

    # CNN architecture
    print("training CNN RNN...")

    model = Sequential()
    model.add(Embedding(nb_words, embed_dim,
                        weights=[embedding_matrix], input_length=max_seq_len, trainable=False))
    model.add(SpatialDropout1D(0.2))
    model.add(Conv1D(num_filters, 3, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))
    # model.add(Bidirectional(GRU(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Dense(num_classes, activation='sigmoid'))  # multi-label (k-hot encoding)

    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    # checkpoint
    filepath = output_dir
    checkpoint = ModelCheckpoint(filepath + "best.h5", monitor='val_accuracy', verbose=1, save_best_only=True,
                                 mode='max')
    model.summary()
    # define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=5, verbose=1)
    callbacks_list = [early_stopping, checkpoint]

    # to categorical TO DO

    Y_train = utils.to_categorical(Y_train, num_classes)
    # model training
    hist = model.fit(X_train_word_seq, Y_train, batch_size=batch_size, epochs=num_epochs, callbacks=callbacks_list,
                     validation_split=0.2, shuffle=True, verbose=2)
    print('X_test_word_seq.shape', X_test_word_seq.shape)
    Y_predict = model.predict(X_test_word_seq)
    print('Y_predict', Y_predict)
    np.save(str(output_dir) + 'Y_predict.npy', Y_predict)
    model.save(str(output_dir) + 'model' + str(job_number))

    return Y_predict


###############################################
def DPCNN_architecture(X_train_word_seq, X_test_word_seq, Y_train, max_seq_len, nb_words, embedding_matrix,
                       embed_dim, output_dir, job_number, batch_size, num_epochs, num_filters, weight_decay):
    num_classes = np.max(Y_train) + 1

    # CNN architecture
    print("training DP CNN...")

    model = Sequential()
    model.add(Embedding(nb_words, embed_dim,
                        weights=[embedding_matrix], input_length=max_seq_len, trainable=False))
    model.add(SpatialDropout1D(0.2))
    repeat = 3
    size = num_classes
    model.add(Conv1D(filters=250, kernel_size=3, padding='same', strides=1))
    model.add(Activation(activation='relu'))
    model.add(Conv1D(filters=250, kernel_size=3, padding='same', strides=1))
    model.add(Activation(activation='relu'))
    model.add(Conv1D(filters=250, kernel_size=3, padding='same', strides=1))

    # for _ in range(repeat):
    # model.add(MaxPooling1D(pool_size=3, strides=2, padding='same'))
    # size = int((size + 1) / 2)
    # model.add(Activation(activation='relu'))
    # model.add(Conv1D(filters=250, kernel_size=3, padding='same', strides=1))
    # model.add(Activation(activation='relu'))
    # model.add(Conv1D(filters=250, kernel_size=3, padding='same', strides=1))

    # model.add(MaxPooling1D(pool_size=size))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='sigmoid'))  # multi-label (k-hot encoding)

    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    # checkpoint
    filepath = output_dir
    checkpoint = ModelCheckpoint(filepath + "best.h5", monitor='val_accuracy', verbose=1, save_best_only=True,
                                 mode='max')
    model.summary()
    # define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=5, verbose=1)
    callbacks_list = [early_stopping, checkpoint]

    # to categorical TO DO

    Y_train = utils.to_categorical(Y_train, num_classes)
    # model training
    hist = model.fit(X_train_word_seq, Y_train, batch_size=batch_size, epochs=num_epochs, callbacks=callbacks_list,
                     validation_split=0.2, shuffle=True, verbose=2)
    print('X_test_word_seq.shape', X_test_word_seq.shape)
    Y_predict = model.predict(X_test_word_seq)
    print('Y_predict', Y_predict)
    np.save(str(output_dir) + 'Y_predict.npy', Y_predict)
    model.save(str(output_dir) + 'model' + str(job_number))

    return Y_predict
