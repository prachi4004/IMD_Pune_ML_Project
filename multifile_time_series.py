import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

yr =input("Enter the year you want to see the time series for:")
filename="./NCEP_Reanalysis2_data/2m_Temp/air.2m.gauss." + yr + ".nc"
da=xr.open_dataset(filename)
print(da)


############Read the individual variables in the dataset########
air=da.air
lat=da.lat
lon=da.lon
#print(lon)
new_air=air-(273.5)  



################ Subsetting in all dimensions###################
#da1=da.sel(time='2020-01-01',lat=slice(20.5,7.5),lon=slice(68.0,90.5))
year=int(yr)
da1 = da.sel(lat='18.5204', lon='73.8567', method='nearest')
date1 = datetime(year, 1, 1, 0, 0, 0, 0)
date2 = datetime(year, 12, 31, 23, 59, 59, 999999)
da1 = da1.sel(time=slice(date1, date2))
new_da1 = da1.copy()  # Make a copy of the subsetted data array
new_da1['air'] = new_air.sel(lat='18.5204', lon='73.8567', method='nearest').sel(time=slice(date1, date2))# Replace 'air' variable with temperature in Celsius

#new_da1['air'] = new_air.sel(lat='18.5204', lon='73.8567', method='nearest') 
#new_da1 = new_da1.sel(time=slice(date1, date2))

print(new_da1)
print(new_da1.air)
plot=new_da1.air.plot()
plt.title(f"Daily Forecast of Air Temperature (°C) at 2m for the year {year}")  # Add the title to the plot
plt.ylabel("Air Temperature (°C)")  # Set the y-axis label

plt.savefig("timeseries.png")

###################  Time Series  #########################
# Access the data from RF_final_time
rfl = RF_final_time['rfl']
time = RF_final_time['time']

# Create the plot
plt.plot(time, rfl)

# Set the labels for x and y axes
plt.xlabel('Time')
plt.ylabel('Rainfall (mm)')

# Set the title of the plot
plt.title('Rainfall Timeseries')

# Display the plot
plt.savefig("timeseriesR.png")