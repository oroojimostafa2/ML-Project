import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.regression import *

# Targets (First: Oil and Downhole Temperature Production)

df = pd.read_csv("Volve.csv", index_col=0)

# Processing the Data for Oil Production
# 1. At the begigning on stream hours is zero. Remove this Section.
df = df.loc["2007-07-27":]

# 2. We are divind the data to before and afert P, T sensor got Broken (2010-01-14)
df1 = df.loc[:"2010-01-14"]
df2 = df.loc["2010-01-14":]

# 3. Training Models using Pycaret for Oil Production

# 3.1. for df1 / BORE_GAS_VAL must be removed.

s1 = setup(data=df1.drop("BORE_GAS_VOL", axis=1), target="BORE_OIL_VOL", session_id="1")
best1 = compare_models()
plot_model(best1, plot = 'residuals')
plot_model(best1, plot = 'feature')

s2 = setup(data=df2.drop(["BORE_GAS_VOL", "AVG_DOWNHOLE_TEMPERATURE", "AVG_DOWNHOLE_PRESSURE"],\
                          axis=1), target="BORE_OIL_VOL", session_id="2")
best2 = compare_models()
plot_model(best2, plot = 'residuals')
plot_model(best2, plot = 'feature')

