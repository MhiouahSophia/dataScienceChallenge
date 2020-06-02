from configparser import ConfigParser
from cleaning_data import Cleaning_text
from split_data import split_train_test
from processing_data import word_count, tfidf
import pandas as pd
import json
import shutil
import os
import pickle


def main():
    # parser config
    config_file = "..//config.ini"
    cp = ConfigParser()
    cp.read(config_file)

    # default config
    output_dir = cp["DATA"].get("output_dir")
    data_path = cp["DATA"].get("data_path")
    remove_nan = cp["DATA"].get("False")

    feedback_clean, theme_clean, theme_code_clean = Cleaning_text(data_path, remove_nan)
    print(feedback_clean[0])
    X_train, X_test, Y_train, Y_test = split_train_test(feedback_clean, theme_code_clean)
    print(X_train.head())
    X_train, X_test = tfidf(X_train),  tfidf(X_test)

    print(' ******** Loaded, cleaned and split the data')
    print(X_train[1])


if __name__ == "__main__":
    main()
