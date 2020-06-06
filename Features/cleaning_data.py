# module to clean
import pandas as pd
# package allows to remove puncutation in text
import string
import nltk
import re
# package to retrieve stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('word_tokenize')
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import FrenchStemmer
# package allows to
nltk.download('punkt')
# package for Lemmatize the text (take only the root)
from nltk.stem import WordNetLemmatizer
# CountVectorizer for word embeddings - word count
from sklearn.feature_extraction.text import CountVectorizer
from spellchecker import SpellChecker



def fr_stop_word():
    # List of stop word
    stopWords = set(stopwords.words('french'))
    stopWords = [i for i in stopWords]
    return stopWords


def load_data(path):
    return pd.read_csv(path, encoding='utf-8')


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


def cleaning_text(path, remove_nan=False):
    train_data = load_data(path)
    if remove_nan:
        # Clean data without NaN df_feedback
        df_feedback = train_data[["Q", "Q_1 Thème"]][train_data["Q"].notnull()]
        df_feedback['Q_1_Thème_code'] = df_feedback["Q_1 Thème"].astype('category').cat.codes

    else:
        train_data['Q'] = train_data['Q'].fillna({'Q': 'NaN'}).astype(str)
        train_data['Q_1 Thème'] = train_data['Q_1 Thème'].fillna({'Q_1 Thème': 'NaN'}).astype(str)
        df_feedback = train_data[["Q", "Q_1 Thème"]].copy()
        df_feedback.loc[:, 'Q_1_Thème_code'] = df_feedback["Q_1 Thème"].astype('category').cat.codes.values

    # remove special character, symbols and number
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    STOPWORDS = set(stopwords.words('french'))

    def clean_text(text):
        """
            text: a string
            return: modified initial string
        """
        text = REPLACE_BY_SPACE_RE.sub(' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
        text = BAD_SYMBOLS_RE.sub('', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
        text = text.lower()  # lowercase text
        text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # delete stopwors from text
        return text

    feedback_list_clean = df_feedback['Q'].apply(clean_text).to_list()
    # tokenize sentences
    spell = SpellChecker(language='fr')
    feedback_tokens = [word_tokenize(w)for i, w in enumerate(feedback_list_clean)]
    correct_spelling = []
    for doc in feedback_tokens:
        for word in doc:
            correct_spelling.append(spell.correction(word))
    feedback_tokens = [correct_spelling]

    # remove word less than 2 caracters
    feed_clean = []
    for i in range(0, len(feedback_tokens)):
        s = []
        for word in feedback_tokens[i]:
            if len(word) > 2:
                s.append(word)
            # if len(word)>1:
            # s=["".join(word)]
        feed_clean.append(" ".join(s))

    df_feedback.loc[:, 'Q_clean'] = feed_clean

    df_feedback_clean = df_feedback[df_feedback['Q_clean'].apply(lambda x: len(x) > 2)]

    return df_feedback_clean



