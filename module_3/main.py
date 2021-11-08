import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)


def parse_cuisine(x):
    if not pd.isna(x):
        x = [i.strip() for i in x[1: len(x) - 1].split(', ')]
    return x


db = pd.read_csv('main_task_new.csv')

print(db.info())

dum_city = pd.get_dummies(db['City'])
recent_price = db['Price Range'].mode()[0]
db['Price Range'] = db['Price Range'].fillna(db['Price Range'].mode()[0])

dum_price_range = pd.get_dummies(db['Price Range'])

db['Cuisine Style'] = db['Cuisine Style'].apply(parse_cuisine)

print(db.info())
# print(db.head())
# count_cuisine = db.explode(['Cuisine Style'])['Cuisine Style'].value_counts().idxmax()
count_cuisine = db.explode(['Cuisine Style'])['Cuisine Style'].mode()[0]
print(count_cuisine)
# db['Cuisine Style'] = db['Cuisine Style'].fillna(count_cuisine)
# c = pd.get_dummies(db['Cuisine Style'].apply(pd.Series).stack()).sum(level=0)
dum_cuisine_style = pd.get_dummies(db['Cuisine Style'].explode()).groupby(level=0).sum()
# print(pd.get_dummies(db['Cuisine Style'].explode()).groupby(level=0).sum())
# print(c)

print(db['ID_TA'])

