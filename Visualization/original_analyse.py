import load_data
import numpy as np

#Package allows to make a count of customer feedback according each theme
from collections import Counter

#------------------------------------
# ********* Plot *****************
#------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns


###############################

train_data[train_data["Q"].isnull()]["Q_1 Thème"].value_counts()


###############################


# Create class distribution
Class = Counter(theme_code)
Class = sorted(Class.items(), key=lambda x: x[0])
Class = dict(Class)

height = Class.values()
bars = tuple(Class.keys())

data_class = pd.DataFrame(zip(bars, height), columns=['Class', 'Class_Count'])

plt.figure(figsize=(15, 8))

sns.barplot(x='Class', y='Class_Count', data=data_class,
            color="#69b3a2", linewidth=3)

plt.title("Train Class distributions")

plt.show()

###############################

#Dataframe Nomenclature theme to retrieve theme and theme_code together
Nomenclature_Theme=pd.DataFrame(zip(theme_code,theme),columns=["Code","Theme"]).drop_duplicates().sort_values(by=['Code']).reset_index(drop=True)

#Dataframe count()
df_count=pd.DataFrame(Class.items(),columns=["Code","Count"])
Nomenclature_Theme.merge(df_count, on="Code", how="left")