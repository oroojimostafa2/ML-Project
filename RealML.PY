import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("Volve.csv", index_col=0)

# Processing the data

# 1. Removing Nulls
# There is no null in the data
# print(df.isnull().sum())

# 2. Removing Duplicate
# Most of the duplicates are zeros, therefore we are going to start from the time
# that production starts

df = df[df["ON_STREAM_HRS"]>0]

# 3. Removing Outliers !!!!

def count_outliers(df):
    # Removing Outliers

    columns_outlier = dict()
    df_length = df.shape[0]

    for column in df.columns:

        array = np.array(df[column])
        
        q1 = np.percentile(array, 25)
        q3 = np.percentile(array, 75)
        iqr = q3 -q1
        
        upper_limit = q3 + (1.5*iqr)
        lower_limit = q1 - (1.5*iqr)
        
        count = df[(df[column]<lower_limit) | (df[column]>upper_limit)].shape[0]

        columns_outlier[column] = [count * 100 / df_length]
        
    return columns_outlier

# 4. Removing or Replacing Zeros

def count_zeros(df):

    count_zero = dict()

    for column in df.columns:
        count = df[df[column]==0].shape[0]
        count_zero[column] = [count]
    return count_zero


# Dividing the Data to 2 different parts. Before 2010 (When the p and T sensors have gotten broken)

df.index = pd.to_datetime(df.index)
df = df[df["AVG_DOWNHOLE_PRESSURE"] > 0]

X = df[['onstreaminject_HRS', 'BORE_WI_VOL', 'ON_STREAM_HRS', 'AVG_DP_TUBING','AVG_ANNULUS_PRESS',\
        'AVG_CHOKE_SIZE_P in percentage', 'AVG_WHP_P', 'AVG_WHT_P', 'DP_CHOKE_SIZE', 'BORE_OIL_VOL',\
        'BORE_GAS_VOL','BORE_WAT_VOL']]

y = df["AVG_DOWNHOLE_TEMPERATURE"]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20, \
                                                random_state=42)
sc = StandardScaler()
Xtrain_scaled = sc.fit_transform(Xtrain)
Xtest_scaled = sc.transform(Xtest)

clf = RandomForestRegressor(n_estimators=300)
clf.fit(Xtrain_scaled, ytrain)
y_p = clf.predict(Xtest_scaled)
plt.scatter(y_p, ytest)

print(r2_score(ytest, y_p))


clf2 = LinearRegression()
clf2.fit(Xtrain_scaled, ytrain)
y_p = clf2.predict(Xtest_scaled)
plt.scatter(y_p, ytest)

print(r2_score(ytest, y_p))




