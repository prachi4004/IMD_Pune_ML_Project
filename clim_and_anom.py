#Plotting graphs for anomaly and climatology 
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from datetime import datetime
import xarray as xr
import numpy as np

#Taking input from user
date_s =input("Enter the start date you want to see the graphs for in the form of YYYY-MM-DD:")
date_e =input("Enter the end date you want to see the graphs for in the form of YYYY-MM-DD:")

yr= date_s[:4]

# Convert the date string to a datetime object
date_object_s = datetime.strptime(date_s, "%Y-%m-%d")
date_object_e = datetime.strptime(date_e, "%Y-%m-%d")


# Extract the day number
day_number1 = date_object_s.timetuple().tm_yday
day_number2 = date_object_e.timetuple().tm_yday              # day_number1 and day_number2 can vary from 1 to 366

dayno1=int(day_number1)
#dayno1=dayno1-1             
dayno2=int(day_number2)
#dayno2=dayno2-1              
#print(day_number)

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
# Update the variable for day of the year
ds_new['dayofyear'] = ds_new.time.dt.dayofyear

# Climatology and anomaly

ds_new_climato = ds_new.groupby("dayofyear").mean("time")
print(ds_new_climato)

ds_new_anom = ds_new.groupby("dayofyear") - ds_new.groupby("dayofyear").mean("time")
print(ds_new_anom)

######################### Simple plotting with xarray#################################
fig, ax = plt.subplots()

#   Anomaly plotting
ds_new_anom.air.sel(time=slice(date_object_s,date_object_e), lat=slice(39.0, 6.2), lon=slice(67, 97.5)).mean(axis=0).plot(ax=ax,vmin= -5, vmax = 5,cmap='coolwarm', levels=np.arange(-5, 5, 0.5))
# Read and plot the shapefile
shp = gpd.read_file("./india updated state boundary/india updated state boundary.shp")
shp.plot(ax=ax, alpha=0.8, facecolor='None', lw=1)

plt.savefig("Anomaly_temp_period.png")

#   Climatology plotting

ds_new_climato.air.sel(dayofyear=slice(dayno1 ,dayno2), lat=slice(39.0, 6.2), lon=slice(67, 97.5)).mean(axis=0).plot(ax=ax, levels=np.arange(270, 310, 0.5))
# Read and plot the shapefile
shp = gpd.read_file("./india updated state boundary/india updated state boundary.shp")
shp.plot(ax=ax, alpha=0.8, facecolor='None', lw=1)

# Add text description
description = f' Climatology for the time period {date_s} to {date_e}, for the year {yr} from day number {dayno1} to {dayno2}'
plt.text(0.5, 1.09, description, transform=ax.transAxes, ha='center', fontsize=6)

plt.savefig("Climatology_temp_period.png")

