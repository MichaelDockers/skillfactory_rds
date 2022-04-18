# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Making all trains and test similar and compatible
#
# Rework of [this](2022-03-19_train-test_comparison.ipynb) notebook

# %% [markdown]
# ## imports and reading data

# %%
from warnings import filterwarnings
from datetime import datetime

import re
import json
import time

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from lib.data_viz_functions import *

filterwarnings("ignore")
sns.set()

# %% [markdown] tags=[]
# ## reading raw data

# %%
train_jane = pd.read_pickle("data/train_df_full_part1.pkl.zip", compression="zip")
train_sokolov = pd.read_pickle("data/all_auto_ru_09_09_2020.pkl.zip", compression="zip")
test = pd.read_pickle("data/test.pkl.zip", compression="zip")

# %% [markdown]
# ## viewing NAs

# %%
plt.figure(figsize=(16, 5))
sns.heatmap(train_jane.drop_duplicates(subset=["car_url"]).isna(), cmap="viridis")
plt.show()

# %%
plt.figure(figsize=(16, 5))
sns.heatmap(train_sokolov.drop_duplicates().isna(), cmap="viridis")
plt.show()

# %%
plt.figure(figsize=(16, 5))
sns.heatmap(test.drop_duplicates(subset=["car_url"]).isna(), cmap="viridis")
plt.show()

# %% [markdown]
# ## functions and vocs for preprocessing

# %%
colors_dict = {
    "040001": "чёрный",
    "EE1D19": "красный",
    "0000CC": "синий",
    "CACECB": "серый",
    "007F00": "зелёный",
    "FAFBFB": "белый",
    "97948F": "серый",
    "22A0F8": "синий",
    "660099": "фиолетовый",
    "200204": "коричневый",
    "C49648": "коричневый",
    "DEA522": "золотистый",
    "4A2197": "фиолетовый",
    "FFD600": "жёлтый",
    "FF8649": "оранжевый",
    "FFC0CB": "розовый",
}

vendor_voc_additional = {
    "CADILLAC": "AMERICAN",
    "CHERY": "CHINESE",
    "CHEVROLET": "AMERICAN",
    "CHRYSLER": "AMERICAN",
    "CITROEN": "EUROPEAN",
    "DAEWOO": "KOREAN",
    "DODGE": "AMERICAN",
    "FORD": "AMERICAN",
    "GEELY": "CHINESE",
    "HYUNDAI": "KOREAN",
    "JAGUAR": "EUROPEAN",
    "JEEP": "AMERICAN",
    "KIA": "KOREAN",
    "MAZDA": "JAPANESE",
    "MINI": "EUROPEAN",
    "OPEL": "EUROPEAN",
    "PEUGEOT": "EUROPEAN",
    "PORSCHE": "EUROPEAN",
    "RENAULT": "EUROPEAN",
    "SUBARU": "JAPANESE",
    "SUZUKI": "JAPANESE",
    "GREAT_WALL": "CHINESE",
    "LAND_ROVER": "EUROPEAN",
    "SSANG_YONG": "KOREAN",
}

transmission_dict = {
    "роботизированная": "ROBOT",
    "автоматическая": "AUTOMATIC",
    "механическая": "MECHANICAL",
    "вариатор": "VARIATOR",
}


# %%
def parse_ownership_duration(train_str: str) -> int:
    """
    Processing "Владение" column
    Returns ownership duration in days (integer)
    """
    baseline_str = "{'year': 2020, 'month': 9}"
    if not isinstance(train_str, str):
        return np.nan
    elif "year" in str(train_str):
        baseline_dict = json.loads(baseline_str.replace("'", '"'))
        train_dict = json.loads(train_str.replace("'", '"'))
        baseline_date = datetime.strptime(
            f"{baseline_dict['year']}-{baseline_dict['month']}-1", "%Y-%m-%d"
        )
        train_date = datetime.strptime(
            f"{train_dict['year']}-{train_dict['month']}-1", "%Y-%m-%d"
        )
        return (baseline_date - train_date).days
    elif " и " in str(train_str):
        return (
            int(train_str.split(" ")[0]) * 365
            + int(train_str.split(" и ")[1].split(" ")[0]) * 30
        )
    else:
        return int(train_str.split(" ")[0]) * 365


def get_number_of_owners_from_owners(in_str):
    """
    Processing "Владельцы" column
    Extracting numbers from text
    """
    if not isinstance(in_str, str):
        return None
    else:
        result = in_str.replace("\xa0", "")
        return int(re.sub("\D", "", result))


def get_volume_from_vehicleConfiguration(in_str: str) -> str:
    """
    Processing "engineDisplacement"
    Extracting numbers but resurns as str: to convers using .astype method after
    """
    if not pd.isna(in_str):
        if 'electro' in in_str.lower():
            return np.nan
        else:
            engine_volume = re.findall("[\d]*[.][\d]+", in_str)
            if len(engine_volume) >= 1:
                return engine_volume[0]
            else:
                return np.nan
    else:
        return np.nan


def standartize_model_name(in_str) -> str:
    '''
    From 3 data sets standartize model name. Return string
    '''
    if not pd.isna(in_str):
        in_str = in_str.upper()
        in_str = in_str.replace('-', '_')
        in_str = in_str.replace(' ', '_')
        in_str = in_str.replace('_СЕРИИ', 'ER')
        in_str = in_str.replace('_КЛАСС', '_KLASSE')
        in_str = in_str.replace('RAV4', 'RAV_4')
        in_str = in_str.replace('190_(W201)', 'W201')
        in_str = in_str.replace('QASHQAI+2', 'QASHQAI_PLUS_2')
        in_str = in_str.replace('NAVARA_(FRONTIER)', 'NAVARA')
        in_str = in_str.replace('DELICA_D:5', 'DELICA_D_5')
        in_str = in_str.replace('DELICA_D:2', 'DELICA_D2')
        in_str = in_str.replace('ODYSSEY_(NORTH_AMERICA)', 'ODYSSEY')
        in_str = in_str.replace('ODYSSEY_NA', 'ODYSSEY')
        in_str = in_str.replace("R'NESSA", 'RNESSA')
        in_str = in_str.replace('2ACTIVETOURER', '2ER_ACTIVE_TOURER')
        in_str = in_str.replace('2GRANDTOURER', '2ER_GRAN_TOURER')
        in_str = in_str.replace('PASSAT_(NORTH_AMERICA)', 'PASSAT')
        
        return in_str


# %% [markdown] tags=[]
# ## cleanup start

# %% [markdown]
# ### NaN's, duplicated and new cars cleanup

# %%
# drop rows with NaN's in price
train_jane.dropna(subset=["price"], inplace=True)

# drop rows with duplicates in 'car_url' columns
train_jane.drop_duplicates(subset='car_url', inplace=True)

# drop rows with new cars ads 
mask_new_cars = train_jane['car_url'].str.contains('new')
train_jane.drop(train_jane[mask_new_cars].index, inplace=True)

train_jane.reset_index(drop=True, inplace=True)

# %% [markdown]
# ### enginePower, engineDisplacement and name

# %%
train_jane["enginePower"] = train_jane["enginePower"].replace("undefined N12", None)
train_jane["enginePower"] = (
    train_jane[~pd.isna(train_jane["enginePower"])]["enginePower"]
    .str.split()
    .str.get(0)
    .astype("int")
)

# %%
test["enginePower"] = test["enginePower"].replace("undefined N12", None)
test["enginePower"] = (
    test[~pd.isna(test["enginePower"])]["enginePower"]
    .str.split()
    .str.get(0)
    .astype("int")
)

# %%
train_jane["engineDisplacement"] = train_jane["engineDisplacement"].replace(
    " LTR", None
)
train_jane["engineDisplacement"] = (
    train_jane[~pd.isna(train_jane["engineDisplacement"])]["engineDisplacement"]
    .str.split()
    .str.get(0)
    .astype("float")
)

# %%
train_sokolov["engineDisplacement"] = (
    train_sokolov["name"].apply(get_volume_from_vehicleConfiguration).astype("float64")
)

# %%
test["engineDisplacement"] = test["engineDisplacement"].replace(" LTR", None)
test["engineDisplacement"] = (
    test[~pd.isna(test["engineDisplacement"])]["engineDisplacement"]
    .str.split()
    .str.get(0)
    .astype("float")
)

# %%
test["Владельцы"] = test["Владельцы"].apply(get_number_of_owners_from_owners)
train_jane["Владельцы"] = train_jane["Владельцы"].apply(
    get_number_of_owners_from_owners
)

# %%
# train_jane["model_name"] = train_jane.model_name.apply(
#     lambda x: x.upper() if not pd.isna(x) else x
# )
# test["model_name"] = test.model_name.apply(lambda x: x.upper() if not pd.isna(x) else x)

# %%
brand_set = set(train_jane['brand'].unique()) | set(train_sokolov['brand'].unique()) | set(test['brand'].unique())

# %% [markdown] tags=[]
# ### model_name

# %%
# One column name for all data sets
train_sokolov.rename(columns={'model': 'model_name'}, inplace=True)

# Applying func to all data sets
train_jane['model_name'] = train_jane['model_name'].apply(standartize_model_name)
train_sokolov['model_name'] = train_sokolov['model_name'].apply(standartize_model_name)
test['model_name'] = test['model_name'].apply(standartize_model_name)

# %%
# for brand in brand_set:
#     all_models = (
#         set(train_jane.dropna(subset='model_name')[train_jane['brand'] == brand]['model_name'].unique()) | 
#         set(train_sokolov.dropna(subset='model_name')[train_sokolov['brand'] == brand]['model_name'].unique()) | 
#         set(test.dropna(subset='model_name')[test['brand'] == brand]['model_name'].unique())
#     )
#     print(sorted(all_models))

# %% [markdown]
# ### vendor -- fillna

# %%
vendor_voc = (
    test[["brand", "vendor"]].drop_duplicates().set_index("brand").to_dict()["vendor"]
) | vendor_voc_additional

# %%
train_jane["vendor"] = train_jane["brand"].map(vendor_voc)
train_sokolov["vendor"] = train_sokolov["brand"].map(vendor_voc)

# %% [markdown]
# ### dropna by vendor

# %%
train_jane.dropna(subset=["vendor"], axis="rows", inplace=True)

# %% [markdown]
# ### Parsed data standartize

# %%
train_jane['parsed_date'] = train_jane['parsing_unixtime'].apply(
    lambda x: time.strftime("%Y-%m-%d", time.localtime(x)) if not pd.isna(x) else x
)
test['parsed_date'] = test['parsing_unixtime'].apply(
    lambda x: time.strftime("%Y-%m-%d", time.localtime(x)) if not pd.isna(x) else x
)
train_sokolov['parsed_date'] = '2020-09-09'

# %% [markdown]
# ## deleting cols which can not be used by some reason. At least for now

# %%
del test["car_url"]
del test["complectation_dict"]
del test["equipment_dict"]
del test["image"]
del test["model_info"]
del test["name"]
del test["parsing_unixtime"]
del test["priceCurrency"]
del test["sell_id"]
del test["vehicleConfiguration"]
del test["Состояние"]
del test["Таможня"]
del test["super_gen"]
del train_sokolov["name"]
del train_sokolov["hidden"]
del train_sokolov["start_date"]
del train_sokolov["vehicleConfiguration"]
del train_sokolov["Комплектация"]
del train_sokolov["Состояние"]
del train_sokolov["Таможня"]
del train_jane["car_url"]
del train_jane["complectation_dict"]
del train_jane["date_added"]
del train_jane["equipment_dict"]
del train_jane["image"]
del train_jane["model_info"]
del train_jane["name"]
del train_jane["parsing_unixtime"]
del train_jane["priceCurrency"]
del train_jane["region"]
del train_jane["sell_id"]
del train_jane["vehicleConfiguration"]
del train_jane["views"]
del train_jane["Состояние"]
del train_jane["Таможня"]
del train_jane["super_gen"]

# %% [markdown]
# ## appending trains and further cleanup

# %%
train_jane["sample"] = "jane"
train_sokolov["sample"] = "sokolov"
train = pd.concat([train_jane, train_sokolov])

# %%
train.drop_duplicates(inplace=True)
train.dropna(subset=["price"], inplace=True)
train.reset_index(drop=True, inplace=True)

# %%
train.loc[train.bodyType.isna()]

# %%
train.dropna(subset="bodyType", inplace=True)

# %% [markdown]
# ### Владение

# %%
train["Владение"] = train["Владение"].apply(parse_ownership_duration)
test["Владение"] = test["Владение"].apply(parse_ownership_duration)

# %% [markdown]
# ### bodyType

# %%
train["bodyType"] = train["bodyType"].str.replace("-", " ")
train["bodyType"] = train["bodyType"].str.lower()

# %%
train["bodyType"] = train["bodyType"].str.replace("-", " ")
test["bodyType"] = test["bodyType"].str.replace("-", " ")

# %%
train["bodyType"] = train["bodyType"].apply(
    lambda x: x.split()[0] if not pd.isna(x) else x
)
test["bodyType"] = test["bodyType"].apply(
    lambda x: x.split()[0] if not pd.isna(x) else x
)

# %% [markdown]
# ### color

# %%
train["color"].replace(colors_dict, inplace=True)

# %% [markdown]
# ### vehicleTransmission

# %%
test["vehicleTransmission"] = test["vehicleTransmission"].replace(transmission_dict)
train["vehicleTransmission"] = train["vehicleTransmission"].replace(transmission_dict)

# %% [markdown]
# ### ПТС and Руль

# %%
train["ПТС"].replace({"ORIGINAL": "Оригинал", "DUPLICATE": "Дубликат"}, inplace=True)
train["Руль"].replace({"LEFT": "Левый", "RIGHT": "Правый"}, inplace=True)

# %% [markdown]
# ### description

# %%
train["description"] = train["description"].str.lower()
test["description"] = test["description"].str.lower()

# %%
r = re.compile("[\W_]+")

# %%
train["description"] = train["description"].apply(
    lambda x: r.sub(" ", x) if not pd.isna(x) else x
)
test["description"] = test["description"].apply(
    lambda x: r.sub(" ", x) if not pd.isna(x) else x
)

# %% [markdown]
# ### converting types

# %%
train.loc[train["bodyType"].isna() == False, "bodyType"] = train.loc[train["bodyType"].isna() == False]["bodyType"].astype("str")
train.loc[train["brand"].isna() == False, "brand"] = train.loc[train["brand"].isna() == False]["brand"].astype("str")
train.loc[train["color"].isna() == False, "color"] = train.loc[train["color"].isna() == False]["color"].astype("str")
train.loc[train["description"].isna() == False, "description"] = train.loc[train["description"].isna() == False]["description"].astype("str")
train.loc[train["engineDisplacement"].isna() == False, "engineDisplacement"] = train.loc[train["engineDisplacement"].isna() == False]["engineDisplacement"].astype("float32")
train.loc[train["enginePower"].isna() == False, "enginePower"] = train.loc[train["enginePower"].isna() == False]["enginePower"].astype("float32")
train.loc[train["fuelType"].isna() == False, "fuelType"] = train.loc[train["fuelType"].isna() == False]["fuelType"].astype("str")
train.loc[train["mileage"].isna() == False, "mileage"] = train.loc[train["mileage"].isna() == False]["mileage"].astype("float32")
train.loc[train["modelDate"].isna() == False, "modelDate"] = train.loc[train["modelDate"].isna() == False]["modelDate"].astype("float32")
train.loc[train["numberOfDoors"].isna() == False, "numberOfDoors"] = train.loc[train["numberOfDoors"].isna() == False]["numberOfDoors"].astype("float32")
train.loc[train["productionDate"].isna() == False, "productionDate"] = train.loc[train["productionDate"].isna() == False]["productionDate"].astype("float32")
train.loc[train["vehicleTransmission"].isna() == False, "vehicleTransmission"] = train.loc[train["vehicleTransmission"].isna() == False]["vehicleTransmission"].astype("str")
train.loc[train["vendor"].isna() == False, "vendor"] = train.loc[train["vendor"].isna() == False]["vendor"].astype("str")
train.loc[train["Владельцы"].isna() == False, "Владельцы"] = train.loc[train["Владельцы"].isna() == False]["Владельцы"].astype("float32")
train.loc[train["Владение"].isna() == False, "Владение"] = train.loc[train["Владение"].isna() == False]["Владение"].astype("float32")
train.loc[train["ПТС"].isna() == False, "ПТС"] = train.loc[train["ПТС"].isna() == False]["ПТС"].astype("str")
train.loc[train["Привод"].isna() == False, "Привод"] = train.loc[train["Привод"].isna() == False]["Привод"].astype("str")
train.loc[train["Руль"].isna() == False, "Руль"] = train.loc[train["Руль"].isna() == False]["Руль"].astype("str")

train.loc[train["price"].isna() == False, "price"] = train.loc[train["price"].isna() == False]["price"].astype("float64")

test.loc[test["bodyType"].isna() == False, "bodyType"] = test.loc[test["bodyType"].isna() == False]["bodyType"].astype("str")
test.loc[test["brand"].isna() == False, "brand"] = test.loc[test["brand"].isna() == False]["brand"].astype("str")
test.loc[test["color"].isna() == False, "color"] = test.loc[test["color"].isna() == False]["color"].astype("str")
test.loc[test["description"].isna() == False, "description"] = test.loc[test["description"].isna() == False]["description"].astype("str")
test.loc[test["engineDisplacement"].isna() == False, "engineDisplacement"] = test.loc[test["engineDisplacement"].isna() == False]["engineDisplacement"].astype("float32")
test.loc[test["enginePower"].isna() == False, "enginePower"] = test.loc[test["enginePower"].isna() == False]["enginePower"].astype("float32")
test.loc[test["fuelType"].isna() == False, "fuelType"] = test.loc[test["fuelType"].isna() == False]["fuelType"].astype("str")
test.loc[test["mileage"].isna() == False, "mileage"] = test.loc[test["mileage"].isna() == False]["mileage"].astype("float32")
test.loc[test["modelDate"].isna() == False, "modelDate"] = test.loc[test["modelDate"].isna() == False]["modelDate"].astype("float32")
test.loc[test["numberOfDoors"].isna() == False, "numberOfDoors"] = test.loc[test["numberOfDoors"].isna() == False]["numberOfDoors"].astype("float32")
test.loc[test["productionDate"].isna() == False, "productionDate"] = test.loc[test["productionDate"].isna() == False]["productionDate"].astype("float32")
test.loc[test["vehicleTransmission"].isna() == False, "vehicleTransmission"] = test.loc[test["vehicleTransmission"].isna() == False]["vehicleTransmission"].astype("str")
test.loc[test["vendor"].isna() == False, "vendor"] = test.loc[test["vendor"].isna() == False]["vendor"].astype("str")
test.loc[test["Владельцы"].isna() == False, "Владельцы"] = test.loc[test["Владельцы"].isna() == False]["Владельцы"].astype("float32")
test.loc[test["Владение"].isna() == False, "Владение"] = test.loc[test["Владение"].isna() == False]["Владение"].astype("float32")
test.loc[test["ПТС"].isna() == False, "ПТС"] = test.loc[test["ПТС"].isna() == False]["ПТС"].astype("str")
test.loc[test["Привод"].isna() == False, "Привод"] = test.loc[test["Привод"].isna() == False]["Привод"].astype("str")
test.loc[test["Руль"].isna() == False, "Руль"] = test.loc[test["Руль"].isna() == False]["Руль"].astype("str")

# %%
train.columns.tolist()

# %% [markdown]
# ## reviewing before write

# %%
train[train.columns.sort_values().tolist()].sample(5, random_state=42).T

# %%
test[test.columns.sort_values().tolist()].sample(5, random_state=42).T

# %% [markdown]
# ### bodyType

# %%
train.bodyType.value_counts(dropna=False)

# %%
test.bodyType.value_counts(dropna=False)

# %% [markdown]
# ### brand

# %%
train.brand.value_counts(dropna=False)

# %%
test.brand.value_counts(dropna=False)

# %% [markdown]
# ### color

# %%
train.color.value_counts(dropna=False)

# %%
test.color.value_counts(dropna=False)

# %% [markdown]
# ### description

# %%
train.description[42]

# %% [markdown]
# ### engineDisplacement

# %%
train.engineDisplacement.value_counts(dropna=False)

# %%
test.engineDisplacement.value_counts(dropna=False)

# %% [markdown]
# ### enginePower

# %%
train.enginePower.value_counts(dropna=False)

# %%
test.enginePower.value_counts(dropna=False)

# %% [markdown]
# ### fuelType

# %%
train.fuelType.value_counts(dropna=False)

# %%
test.fuelType.value_counts(dropna=False)

# %% [markdown]
# ### mileage

# %%
train.mileage.value_counts(dropna=False)

# %%
test.mileage.value_counts(dropna=False)

# %% [markdown]
# ### model_name

# %%
train["model_name"].value_counts(dropna=False)

# %%
test["model_name"].value_counts(dropna=False)

# %% [markdown]
# ### modelDate

# %%
train.modelDate.value_counts(dropna=False)

# %%
test.modelDate.value_counts(dropna=False)

# %% [markdown]
# ### numberOfDoors

# %%
train.numberOfDoors.value_counts(dropna=False)

# %%
test.numberOfDoors.value_counts(dropna=False)

# %% [markdown]
# ### productionDate

# %%
train.productionDate.value_counts(dropna=False)

# %%
test.productionDate.value_counts(dropna=False)

# %% [markdown]
# ### vehicleTransmission

# %%
train.vehicleTransmission.value_counts(dropna=False)

# %%
test.vehicleTransmission.value_counts(dropna=False)

# %% [markdown]
# ### vendor

# %%
train.vendor.value_counts(dropna=False)

# %%
test.vendor.value_counts(dropna=False)

# %% [markdown]
# ### Владельцы

# %%
train.Владельцы.value_counts(dropna=False)

# %%
test.Владельцы.value_counts(dropna=False)

# %% [markdown]
# ### Владение

# %%
train.Владение.value_counts(dropna=False)

# %%
test.Владение.value_counts(dropna=False)

# %% [markdown]
# ### ПТС

# %%
train.ПТС.value_counts(dropna=False)

# %%
test.ПТС.value_counts(dropna=False)

# %% [markdown]
# ### Привод

# %%
train.Привод.value_counts(dropna=False)

# %%
test.Привод.value_counts(dropna=False)

# %% [markdown]
# ### Руль

# %%
train.Руль.value_counts(dropna=False)

# %%
test.Руль.value_counts(dropna=False)

# %% [markdown]
# ### price

# %%
show_IQR(train.query("price < 10_000_000").price)

# %%
show_boxplots(train.query("price < 10_000_000"), "sample", "price")

# %% [markdown]
# ## writing results to files

# %%
train.to_pickle("data/2022-04-06_train_preprocessed.pkl.zip", compression="zip")
test.to_pickle("data/2022-04-06_test_preprocessed.pkl.zip", compression="zip")

# %% [markdown]
# Next: [EDA](2022-03-31_train-test_EDA.ipynb)
