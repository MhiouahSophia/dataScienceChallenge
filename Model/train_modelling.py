from configparser import ConfigParser
from cleaning_data import remove_nan
from split_data import split_train_test
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

    feedback_clean, theme_clean, theme_code_clean = remove_nan(data_path)


    X_train, X_test, Y_train, Y_test = split_train_test(feedback_clean, theme_code_clean)
    print(' ******** Loaded, cleaned and split the data')


if __name__ == "__main__":
    main()
