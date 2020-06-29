import fasttext
from tqdm import tqdm

tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
import gensim
from gensim.models.doc2vec import TaggedDocument
import nltk

# package allows to tokenize sentence into words
nltk.download('word_tokenize')
from nltk.tokenize import word_tokenize

nltk.download('wordnet')
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from tqdm import tqdm
from keras.preprocessing import text, sequence
from keras import utils


def fastText_process(X_train, X_test, output_dir):
    MAX_NB_WORDS = 100000
    max_seq_len = max(X_train.apply(lambda x: len(x.split(' '))))
    print('max_seq_len', max_seq_len)
    embed_dim = 300
    # Load the model and Retrieve 100 dimensions instead of 300 dimensions
    ft = fasttext.load_model('../FastText/cc.fr.300.bin')
    ft.get_dimension()

    # Retrieve FastText vocabulary
    vocab_ft = ft.get_words()

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True, char_level=False)
    print(X_train.head())
    tokenizer.fit_on_texts(X_train)  # leaky
    word_seq_train = tokenizer.texts_to_sequences(X_train)
    word_seq_test = tokenizer.texts_to_sequences(X_test)
    word_index = tokenizer.word_index
    print("dictionary size: ", len(word_index))

    nb_words = min(MAX_NB_WORDS, len(word_index) + 1)
    print("tokenizing input data...")

    embedding_matrix = np.zeros((nb_words, embed_dim))
    # create a weight matrix for words in training docs
    embedding_matrix = np.zeros((nb_words, embed_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = ft.get_sentence_vector(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    embedding_matrix = np.zeros((nb_words, embed_dim))
    words_not_found = []
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = ft.get_sentence_vector(word)
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            words_not_found.append(word)

    print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    # print("sample words not found: ", np.random.choice(words_not_found, 20))
    print('number of word not found', len(words_not_found))

    X_train_word_seq = sequence.pad_sequences(word_seq_train, maxlen=max_seq_len)
    X_test_word_seq = sequence.pad_sequences(word_seq_test, maxlen=max_seq_len)

    return X_train_word_seq, X_test_word_seq, max_seq_len, nb_words, embedding_matrix, embed_dim


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

