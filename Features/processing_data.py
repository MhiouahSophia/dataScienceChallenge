import nltk

nltk.download('wordnet')
# package allows to tokenize sentence into words
nltk.download('word_tokenize')
from nltk.tokenize import word_tokenize
import pandas as pd

# package to retrieve stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
# package allows to stem the text (retrieve only the root)
from nltk.stem.snowball import FrenchStemmer
# package allows to
nltk.download('punkt')
# package for Lemmatize the text (take only the root)
from nltk.stem import WordNetLemmatizer
# CountVectorizer for word embeddings - word count
from sklearn.feature_extraction.text import CountVectorizer
# TfidfVectorizer for word embeddings - tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
import fasttext
from fasttext import util
import os.path


def fr_stop_word():
    # List of stop word
    stopWords = set(stopwords.words('french'))
    stopWords = [i for i in stopWords]
    return stopWords


def stemmed_words(doc):
    # Create a stemmer function in order to keep only root words (racines)
    stemmer = FrenchStemmer()
    analyzer = CountVectorizer().build_analyzer()

    return (stemmer.stem(w) for w in analyzer(doc))


class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


# word count
def word_count(df_feedback):
    stopWords = fr_stop_word()
    vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), analyzer=stemmed_words, max_features=1000,
                                 stop_words=stopWords, strip_accents='ascii')
    return vectorizer.fit_transform(df_feedback).toarray()


# tfidf

def tfidf(df_feedback):
    stopWords = fr_stop_word()
    vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), analyzer=stemmed_words, max_features=1000,
                                 stop_words=stopWords, strip_accents='ascii')
    return vectorizer.fit_transform(df_feedback).toarray()


# fastText

def word2vec_fastText(df_feedback):
    # Load the model and Retrieve 100 dimensions instead of 300 dimensions
    ft = fasttext.load_model('../../FastText/cc.fr.10.bin')
    ft.get_dimension()

    # Retrieve FastText vocabulary
    vocab_ft = ft.get_words()

    # ft.get_sentence_vector('syndic')
    # df_feedback_train_clean[1]

    # Mapping between the word in the corpus vs FastText vocab
    df_feedback_tovec = [[ft.get_sentence_vector(word) for word in word_tokenize(x) if word in vocab_ft]
                                      for x in df_feedback]
    print(df_feedback_tovec.shape)
    return df_feedback_tovec
