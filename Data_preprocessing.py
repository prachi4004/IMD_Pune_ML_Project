import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import xarray as xr
import numpy as np
from numpy import array


############################################### Data Pre-Processing ###########################################################
df=pd.read_csv('Panjim_Data.csv')
year=df['Year']
#print(year)
rf=[]
for i in range(0,year.shape[0]):
    if(year[i]%4==0):
        for j in range(1, 367):                     #loop over all columns named 1 to 366
            rf.append(df[str(j)][i])
    else:
        rf_tmp = list(df.loc[i, '1':'366'])
        del rf_tmp[59]  # Remove the 60th day
        rf.extend(rf_tmp) 
        
#print(rf)

RF=np.array(rf)      #single dimension rf data for Panjim
Time = pd.date_range(start='1971-01-01', end='2020-12-31', freq='D')
RF_da = xr.Dataset(data_vars={'rain': RF}, coords={'time': Time})
#print(RF_da)

# Netcdf data

RF_gridded=xr.open_mfdataset("./Rainfall_Data/RF25_ind*_rfp25.nc")
#print(RF_gridded)
RF_panjim=RF_gridded.sel(LATITUDE='15.496777', LONGITUDE='73.827827', method='nearest')  #TIME is till 365 or 366

# filling missing data in .csv from netcdf
RF_final = np.where(np.isnan(RF_da['rain'].values), RF_panjim['RAINFALL'], RF_da['rain'].values)
#print(RF_final)
print(RF_final.shape)

RF_final_time= xr.Dataset(data_vars={'rfl': RF_final}, coords={'time': Time})
print(RF_final_time)
# Save data to CSV file
data_frame = pd.DataFrame({'RF_final': RF_final})
data_frame.to_csv('RF_final.csv', index=False)
print("Data saved to RF_final.csv")