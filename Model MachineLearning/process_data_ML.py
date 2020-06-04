import nltk
from cleaning_data import stemmed_words, LemmaTokenizer, fr_stop_word
nltk.download('wordnet')
import pandas as pd
# package allows to stem the text (retrieve only the root)

# CountVectorizer for word embeddings - word count
from sklearn.feature_extraction.text import CountVectorizer
# TfidfVectorizer for word embeddings - tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer


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
