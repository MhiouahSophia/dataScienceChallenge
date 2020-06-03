# module to clean
import pandas as pd
# package allows to remove puncutation in text
import string
import nltk

nltk.download('word_tokenize')
from nltk.tokenize import word_tokenize


def load_data(path):
    return pd.read_csv(path, encoding='utf-8')




def cleaning_text(path, remove_nan=False):
    train_data = load_data(path)
    if remove_nan:
        # Clean data without NaN df_feedback
        df_feedback = train_data[["Q", "Q_1 Thème"]][train_data["Q"].notnull()]
        df_feedback['Q_1_Thème_code'] = df_feedback["Q_1 Thème"].astype('category').cat.codes

    else:
        train_data["Q"] = train_data["Q"].astype(str)
        train_data["Q_1 Thème"] = train_data["Q_1 Thème"].astype(str)
        df_feedback = train_data[["Q", "Q_1 Thème"]].copy()
        df_feedback.loc[:, 'Q_1_Thème_code'] = df_feedback["Q_1 Thème"].astype('category').cat.codes.values

    # remove the punctuation '!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~'
    feedback_list_nopunctuation = [w.translate(str.maketrans('', '', string.punctuation)) for i, w in enumerate(df_feedback['Q'])]

    # tokenize sentences
    feedback_tokens = [word_tokenize(w) for i, w in enumerate(feedback_list_nopunctuation)]

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



