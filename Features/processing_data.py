import nltk
nltk.download('wordnet')
# package allows to tokenize sentence into words
nltk.download('word_tokenize')
from nltk.tokenize import word_tokenize
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
def word_count(data):
    stopWords = fr_stop_word()
    vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), analyzer=stemmed_words, max_features=1000,
                                 stop_words=stopWords, strip_accents='ascii')
    return vectorizer.fit_transform(data).toarray()


# tfidf

def tfidf(data):
    stopWords = fr_stop_word()
    vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), analyzer=stemmed_words, max_features=1000,
                                 stop_words=stopWords, strip_accents='ascii')
    return vectorizer.fit_transform(data).toarray()
# word2vec

# fastText
