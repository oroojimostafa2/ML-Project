# THis is the first ML project
# The databases is from a Norwegian Oil Field (vpd.csv)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(r"vpd.csv")
df.drop("NPD_WELL_BORE_CODE", axis=1, inplace=True) # There are two different column for well names. We remove the first one

def well_dataframe(well_name):
    """
    This Function receives the well name and filters the dataframe based on the name.
    """

    df_well = df[df["NPD_WELL_BORE_NAME"] == well_name]
    df_well = df_well.reset_index()
    
    if "index" in df_well.columns:
        df_well.drop("index", axis=1, inplace=True)
    else:
        df_well = df_well
        
    df_well["DATEPRD"] = pd.to_datetime(df_well["DATEPRD"])

    return df_well

# Injection WELLs (WE have 2 inj wells)

inj_1 = well_dataframe("15/9-F-4") # This well is injection well. Therefore only WI column and On Streaam Hours are good to use.
# (Line 15 - 33 Interactive code)
inj_1.drop(['AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE', 'AVG_DP_TUBING',\
           'AVG_ANNULUS_PRESS', 'AVG_CHOKE_SIZE_P', 'AVG_CHOKE_UOM', 'AVG_WHP_P',\
                'AVG_WHT_P', 'BORE_OIL_VOL', 'BORE_GAS_VOL', 'BORE_WAT_VOL', 'FLOW_KIND', 'WELL_TYPE',\
                    "DP_CHOKE_SIZE", "NPD_WELL_BORE_NAME"], axis=1, inplace=True)
inj_1.fillna(value=0, inplace=True)


inj_2 = well_dataframe("15/9-F-5") # This well is injection well. Therefore only WI column and On Streaam Hours are good to use.
# (Line 22 - 38 Interactive_office code)
inj_2 = inj_2[inj_2["FLOW_KIND"] == "injection"] # Because at a time this well has worked as production well
inj_2.drop(['AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE', 'AVG_DP_TUBING',\
           'AVG_ANNULUS_PRESS', 'AVG_CHOKE_SIZE_P', 'AVG_CHOKE_UOM', 'AVG_WHP_P',\
                'AVG_WHT_P', 'BORE_OIL_VOL', 'BORE_GAS_VOL', 'BORE_WAT_VOL', 'FLOW_KIND', 'WELL_TYPE',\
                    "DP_CHOKE_SIZE", "NPD_WELL_BORE_NAME"], axis=1, inplace=True)
inj_2.fillna(value=0, inplace=True)

# Now MErging the Injection wells
df_inj = inj_1.merge(inj_2, on="DATEPRD", how="inner").reset_index()

