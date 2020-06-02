# module to clean
import pandas as pd
# package allows to remove puncutation in text
import string
import nltk

nltk.download('word_tokenize')
from nltk.tokenize import word_tokenize


def load_data(path):
    return pd.read_csv(path, encoding='utf-8')


def Cleaning_text(path, remove_nan=False):
    train_data = load_data(path)
    if remove_nan:
        # Clean data without NaN feedback
        feedback = train_data[train_data["Q"].notnull()]["Q"].values.tolist()
        theme = train_data[train_data["Q"].notnull()]["Q_1 Thème"].values.tolist()
        theme_code = pd.Series(theme).astype('category').cat.codes

    else:
        feedback = train_data["Q"].astype(str)
        theme = train_data["Q_1 Thème"].astype(str)
        theme_code = theme.astype('category').cat.codes

    # remove the punctuation '!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~'
    feedback = [w.translate(str.maketrans('', '', string.punctuation)) for i, w in enumerate(feedback)]

    # tokenize sentences
    tokens = [word_tokenize(w) for i, w in enumerate(feedback)]

    # remove word less than 2 caracters
    feed_clean = []
    for i in range(0, len(tokens)):
        s = []
        for word in tokens[i]:
            if len(word) > 2:
                s.append(word)
            # if len(word)>1:
            # s=["".join(word)]
        feed_clean.append(" ".join(s))

    # remove empty  document
    feed_clean, feed_clean_index = [y for x, y in enumerate(feed_clean) if len(y) > 2], \
                                   [x for x, y in enumerate(feed_clean) if len(y) > 2]

    theme_code_clean = theme_code[feed_clean_index]

    return feed_clean, theme, theme_code_clean
