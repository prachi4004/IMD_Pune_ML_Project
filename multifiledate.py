import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
#import matplotlib.tri as mtri

import pandas as pd
import numpy as np

##########Read the dataset using xarray##########################
date =input("Enter the date in the format YYYY-MM-DD")


yr= date[:4]
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

print(lon)
new_air=air-(273.5)  #convert temperature from kelvin to celsius

################ Subsetting in all dimensions###################
da1=ds_new.sel(time= date,lat=slice(39.0,6.2),lon=slice(67,97.5))
new_da1 = da1.copy()  # Make a copy of the subsetted data array
new_da1['air'] = new_air.sel(time= date, lat=slice(39.0, 6.2), lon=slice(67, 97.5))  # Replace 'air' variable with temperature in Celsius

# Plotting
fig, ax = plt.subplots()
new_da1.air.plot(ax=ax,cmap='coolwarm', vmin= -30, vmax = 30,levels=np.arange(-30,30,2)) # for green and blue plot
# Read and plot the shapefile
shp = gpd.read_file("./india updated state boundary/india updated state boundary.shp")
shp.plot(ax=ax, alpha=0.8, facecolor='None', lw=1)
# Add text description
description = 'Air Temperature (°C) at 2m'
plt.text(0.5, 1.09, description, transform=ax.transAxes, ha='center', fontsize=12)


plt.savefig(f"temp_plot_centigrade_{date}.png")        #used f string style
