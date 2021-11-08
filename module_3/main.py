import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)


def parse_cuisine(x):
    if not pd.isna(x):
        x = [i.strip().strip("'") for i in x[1: len(x) - 1].split(', ')]
    return x


def parse_date(x):
    if len(x) == 8:
        return []
    else:
        x = x[2: len(x) - 2]
        x = x.replace(']', '')
        x = x.replace('[', '')
        x = x.split(', ')
        return x


db = pd.read_csv('main_task_new.csv')

# Reviews column treat and split it on 4 columns

db['Reviews'] = db['Reviews'].apply(parse_date)

db['RV1_date'] = db['Reviews'].apply(lambda x: '2004-04-21' if x == [] else x[-2].strip("''"))
db['RV2_date'] = db['Reviews'].apply(lambda x: '2004-04-21' if x == [] else x[-1].strip("''"))

db['RV1_date'] = pd.to_datetime(db['RV1_date'].dropna(), errors='coerce')
db['RV2_date'] = pd.to_datetime(db['RV2_date'].dropna(), errors='coerce')

db['RV_date_delta'] = db.apply(lambda x: np.timedelta64(1, 'D').astype(int) if (x['RV1_date'] - x['RV2_date']) =='NaT' else
                               x['RV1_date'] - x['RV2_date'], axis=1)
db['RV_date_delta'] = pd.to_numeric(db['RV_date_delta'].dt.days, downcast='integer')


# Creating dummies for City column

dum_city = pd.get_dummies(db['City'])
db = pd.concat([db, dum_city], axis=1, join='outer')


# Treatment of Price Range column
recent_price = db['Price Range'].mode()[0]
db['Price Range'] = db['Price Range'].fillna(db['Price Range'].mode()[0])


# Creating dummies for Price Range column

dum_price_range = pd.get_dummies(db['Price Range'])
db = pd.concat([db, dum_price_range], axis=1, join='outer')


# Treatment of Cuisine Style column

db['Cuisine Style'] = db['Cuisine Style'].apply(parse_cuisine)

# count_cuisine = db.explode(['Cuisine Style'])['Cuisine Style'].value_counts().idxmax()
count_cuisine = db.explode(['Cuisine Style'])['Cuisine Style'].mode()[0]
db['Cuisine Style'] = db['Cuisine Style'].fillna(count_cuisine)
# c = pd.get_dummies(db['Cuisine Style'].apply(pd.Series).stack()).sum(level=0)


# Creating dummies for Cuisine Style column

dum_cuisine_style = pd.get_dummies(db['Cuisine Style'].explode()).groupby(level=0).sum()
# print(pd.get_dummies(db['Cuisine Style'].explode()).groupby(level=0).sum())
db = pd.concat([db, dum_cuisine_style], axis=1, join='outer')


db['Number of Reviews'] = db['Number of Reviews'].fillna(0)
db['Cuisine_Style_Num'] = db['Cuisine Style'].apply(lambda x: len(x) if x is not None else 1)


print(db['Cuisine Style'])


# for column in db.columns:
#     if db[column].dtype == 'object':
#         db.drop(column, axis=1, inplace=True)
#
#
# db.drop(['RV1_date', 'RV2_date'], axis=1, inplace=True)
#
# db = db.fillna(0)
#
# X = db.drop(['Rating'], axis=1)
# y = db['Rating']
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
#
# regr = RandomForestRegressor(n_estimators=100)
# regr.fit(X_train, y_train)
# y_pred = regr.predict(X_test)
#
# print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
