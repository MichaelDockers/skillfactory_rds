import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.core.display import display

data = pd.read_csv('stud_math.csv')

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)

display(data.head(10))
display(pd.DataFrame(data['score']).info())
for col in data.drop(['absences', 'score'], axis=1):
    if data[col].dtype != 'object':
        data[col] = data[col].apply(lambda x: abs(x) if x < 0 else x)

print(data['health'].unique())