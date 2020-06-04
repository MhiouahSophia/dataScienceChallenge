import fasttext
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
import gensim
from gensim.models.doc2vec import TaggedDocument
import re
from fasttext import util
import os.path
# fastText

import itertools
import os
import nltk
# package allows to tokenize sentence into words
nltk.download('word_tokenize')
from nltk.tokenize import word_tokenize
nltk.download('wordnet')
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import numpy as np
import pandas as pd

import keras
from keras import optimizers
from keras import backend as K
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.utils import plot_model
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping

from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import os, re, csv, math, codecs

from keras.preprocessing import text, sequence
from keras import utils


def fastText1_process(X_train, X_test, MAX_NB_WORDS = 100000,
                  embed_dim=300):

    max_seq_len = max(X_train.apply(lambda x: len(x)))
    print('max_seq_len', max_seq_len)

    print("tokenizing input data...")
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True, char_level=False)
    print(X_train.head())
    tokenizer.fit_on_texts(X_train)  # leaky
    word_seq_train = tokenizer.texts_to_sequences(X_train)
    word_seq_test = tokenizer.texts_to_sequences(X_test)
    word_index = tokenizer.word_index
    print("dictionary size: ", len(word_index))

    # pad sequences
    X_train_word_seq = sequence.pad_sequences(word_seq_train, maxlen=max_seq_len)
    X_test_word_seq = sequence.pad_sequences(word_seq_test, maxlen=max_seq_len)

    # load embeddings
    print('loading word embeddings...')
    embeddings_index = {}
    f = codecs.open('../FastText/wiki.simple.vec', encoding='utf-8')
    for line in tqdm(f):
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('found %s word vectors' % len(embeddings_index))

    # embedding matrix
    print('preparing embedding matrix...')
    words_not_found = []
    nb_words = min(MAX_NB_WORDS, len(word_index) + 1)
    embedding_matrix = np.zeros((nb_words, embed_dim))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            words_not_found.append(word)
    print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    print("sample words not found: ", np.random.choice(words_not_found, 20))

    np.save('./Data_processed/X_train_processFastText1.npy', X_train_word_seq)
    np.save('./Data_processed/X_test_processFastText1.npy', X_test_word_seq)

    return X_train_word_seq, X_test_word_seq, max_seq_len, nb_words, embedding_matrix, embed_dim


def fastText2_process(X_train, X_test):
    # Load the model and Retrieve 100 dimensions instead of 300 dimensions
    ft = fasttext.load_model('../../FastText/cc.fr.10.bin')
    ft.get_dimension()

    # Retrieve FastText vocabulary
    vocab_ft = ft.get_words()

    # ft.get_sentence_vector('syndic')
    # df_feedback_train_clean[1]

    # Mapping between the word in the corpus vs FastText vocab
    X_train_tovec = [[ft.get_sentence_vector(word) for word in word_tokenize(x) if word in vocab_ft]
                         for x in X_train]
    X_test_tovec = [[ft.get_sentence_vector(word) for word in word_tokenize(x) if word in vocab_ft]
                         for x in X_test]

    return X_train_tovec, X_test_tovec


def doc2vec_process(X_train, X_test):

    def label_sentences(corpus, label_type):
        """
        Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
        We do this by using the TaggedDocument method. The format will be "TRAIN_i" or "TEST_i" where "i" is
        a dummy index of the post.
        """
        labeled = []
        for i, v in enumerate(corpus):
            label = label_type + '_' + str(i)
            labeled.append(gensim.models.doc2vec.TaggedDocument(v.split(), [label]))
        return labeled

    X_train = label_sentences(X_train, 'X_train')
    X_test = label_sentences(X_test, 'X_train')
    all_data = X_train + X_test
    model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, min_count=1, alpha=0.065, min_alpha=0.065)
    model_dbow.build_vocab([x for x in tqdm(all_data)])

    for epoch in range(30):
        model_dbow.train(utils.shuffle([x for x in tqdm(all_data)]), total_examples=len(all_data), epochs=1)
        model_dbow.alpha -= 0.002
        model_dbow.min_alpha = model_dbow.alpha

    def get_vectors(model, corpus_size, vectors_size, vectors_type):
        """
        Get vectors from trained doc2vec model
        :param doc2vec_model: Trained Doc2Vec model
        :param corpus_size: Size of the data
        :param vectors_size: Size of the embedding vectors
        :param vectors_type: Training or Testing vectors
        :return: list of vectors
        """
        vectors = np.zeros((corpus_size, vectors_size))
        for i in range(0, corpus_size):
            prefix = vectors_type + '_' + str(i)
            vectors[i] = model.docvecs[prefix]
        return vectors

    X_train_vectors_dbow = get_vectors(model_dbow, len(X_train), 300, 'Train')
    X_test_vectors_dbow = get_vectors(model_dbow, len(X_test), 300, 'Test')

    return X_train_vectors_dbow, X_test_vectors_dbow


def dow_keras(X_train, X_test, Y_train, Y_test):
    max_words = 1000
    tokenize = text.Tokenizer(num_words=max_words, char_level=False)
    tokenize.fit_on_texts(X_train)  # only fit on train

    X_train = tokenize.texts_to_matrix(X_train)
    X_test = tokenize.texts_to_matrix(X_test)

    # encoder = LabelEncoder()
    # encoder.fit(X_train)
    # Y_train = encoder.transform(Y_train)
    # Y_test = encoder.transform(Y_test)
    return X_train, X_test

