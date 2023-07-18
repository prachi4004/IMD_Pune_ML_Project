import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

#Taking input from user
date_s =input("Enter the start date you want to see the graphs for in the form of YYYY-MM-DD:")
date_e =input("Enter the end date you want to see the graphs for in the form of YYYY-MM-DD:")

yr= date_s[:4]

# Convert the date string to a datetime object
date_object_s = datetime.strptime(date_s, "%Y-%m-%d")
date_object_e = datetime.strptime(date_e, "%Y-%m-%d")

# List the file paths of the NetCDF files you want to append
file_paths = "./NCEP_Reanalysis2_data/2m_Temp/air.2m.gauss.*.nc"

# Open the dataset with decode_times=False to prevent cftime errors
da = xr.open_mfdataset(file_paths, decode_times=False)            #decode_times=False is done because time coordinate in our netcdf file was not in the standard format ,so we are reading it as a numeric value only.
             
# converting to xarray
xarray_data = xr.DataArray(da.time)
print(xarray_data)
year=int(yr)
time_values = pd.date_range(start='1980-01-01', end='2020-12-31', freq='D')

xarray_data = xr.DataArray(da.time, coords={'time': time_values}, dims='time')
print(xarray_data)
ds = xarray_data
# Create a new dataset with the same variables as 'da' but with the time coordinate from 'ds'
ds_new = da.assign_coords(time=ds.time)

# Verify the new dataset
print(ds_new)

# Access individual variables
air = ds_new.air
lat = ds_new.lat
lon = ds_new.lon
#print(lon)
new_air=air-(273.5)  


################ Subsetting in all dimensions###################
da1 = ds_new.sel(lat='18.5204', lon='73.8567', method='nearest')
da1 = da1.sel(time=slice(date_object_s, date_object_e))
new_da1 = da1.copy()  # Make a copy of the subsetted data array
new_da1['air'] = new_air.sel(lat='18.5204', lon='73.8567', method='nearest').sel(time=slice(date_object_s, date_object_e))# Replace 'air' variable with temperature in Celsius


print(new_da1)
print(new_da1.air)
plot=new_da1.air.plot()
plt.title(f"Daily Forecast of Air Temperature (°C) at 2m for the year {year}")  # Add the title to the plot
plt.ylabel("Air Temperature (°C)")  # Set the y-axis label

plt.savefig("timeseries_output_of_multifile_time_series.png")

