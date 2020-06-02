# module to clean
import pandas as pd


def load_data(path):
    return pd.read_csv(path)

def remove_nan(path):

    train_data = load_data(path)
    #Clean data without NaN feedback
    feedback_clean=train_data[train_data["Q"].notnull()]["Q"].values.tolist()
    theme_clean=train_data[train_data["Q"].notnull()]["Q_1 Thème"].values.tolist()
    theme_code_clean=pd.Series(theme_clean).astype('category').cat.codes

    return feedback_clean, theme_clean, theme_code_clean

