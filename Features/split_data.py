# Scripts to split the data into train and test
import cleaning_data
# Package to split data into train set and validation set
from sklearn.model_selection import train_test_split
import pandas as pd


def split_train_test(feedback_clean, theme_code_clean):
    # theme_train, theme_val = train_test_split(theme_clean, test_size=0.25, shuffle=True, random_state=877839)
    feedback_train, feedback_test, theme_train_code, theme_test_code = train_test_split(feedback_clean,
                                                                                        theme_code_clean,
                                                                                        test_size=0.20, shuffle=True,
                                                                                        random_state=877839)

    # Reset the index for our datasets train and validation
    X_train = pd.DataFrame(feedback_train, columns=["feedback_train"]).reset_index()
    X_train = X_train["feedback_train"]
    # feedback_train=feedback_train.tolist()

    X_test = pd.DataFrame(feedback_test, columns=["feedback_val"]).reset_index()
    X_test = X_test["feedback_val"]
    # feedback_val=feedback_val.tolist()

    Y_train = pd.DataFrame(theme_train_code, columns=["theme_code"]).reset_index()
    Y_train = Y_train["theme_code"]
    # theme_train_code=theme_train_code.tolist()

    Y_test = pd.DataFrame(theme_test_code, columns=["theme_code"]).reset_index()
    Y_test = Y_test["theme_code"]
    # theme_val_code=theme_val_code.tolist()

    return X_train, X_test, Y_train, Y_test
