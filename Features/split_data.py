# Scripts to split the data into train and test
import cleaning_data
# Package to split data into train set and validation set
from sklearn.model_selection import train_test_split
import pandas as pd


def split_train_test(df_feedback_clean, output_dir, job_number):
    # theme_train, theme_val = train_test_split(theme_clean, test_size=0.25, shuffle=True, random_state=877839)
    X_train, X_test, Y_train, Y_test = train_test_split(df_feedback_clean['Q_clean'],
                                                        df_feedback_clean['Q_1_Thème_code'],
                                                        test_size=0.20, shuffle=True,
                                                        random_state=877839)
    X_train.to_csv(str(output_dir) + 'X_train_clean' + str(job_number) + '.csv')
    X_test.to_csv(str(output_dir) + 'X_test_clean' + str(job_number) + '.csv')
    Y_train.to_csv(str(output_dir) + 'Y_train_clean' + str(job_number) + '.csv')
    Y_test.to_csv(str(output_dir) + 'Y_test_clean' + str(job_number) + '.csv')
    return X_train, X_test, Y_train, Y_test
