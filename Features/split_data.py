# Scripts to reading and writing data
import pandas as pd
import cleaning_data

train_data=pd.read_csv('./E&C Customer Feedback - Train.csv')


#theme_train, theme_val = train_test_split(theme_clean, test_size=0.25, shuffle=True, random_state=877839)
feedback_train, feedback_val, theme_train_code, theme_val_code = train_test_split(feedback_clean, theme_code_clean, test_size=0.25, shuffle=True, random_state=877839)

#Reset the index for our datasets train and validation
feedback_train=pd.DataFrame(feedback_train,columns=["feedback_train"]).reset_index()
feedback_train=feedback_train["feedback_train"]
#feedback_train=feedback_train.tolist()

feedback_val=pd.DataFrame(feedback_val,columns=["feedback_val"]).reset_index()
feedback_val=feedback_val["feedback_val"]
#feedback_val=feedback_val.tolist()

theme_train_code=pd.DataFrame(theme_train_code,columns=["theme_code"]).reset_index()
theme_train_code=theme_train_code["theme_code"]
#theme_train_code=theme_train_code.tolist()

theme_val_code=pd.DataFrame(theme_val_code,columns=["theme_code"]).reset_index()
theme_val_code=theme_val_code["theme_code"]
#theme_val_code=theme_val_code.tolist()
