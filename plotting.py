
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
#import matplotlib.tri as mtri

import pandas as pd
import numpy as np

##########Read the dataset using xarray##########################
filename="./NCEP_Reanalysis2_data/2m_Temp/air.2m.gauss.1980.nc"
da=xr.open_dataset(filename)
print(da)

############Read the individual variables in the dataset########
air=da.air
lat=da.lat
lon=da.lon
print(lon)
new_air=air-(273.5)  #convert temperature from kelvin to celsius

################ Subsetting in all dimensions###################
da1=da.sel(time='1980-01-01',lat=slice(39.0,6.2),lon=slice(67,97.5))
new_da1 = da1.copy()  # Make a copy of the subsetted data array
new_da1['air'] = new_air.sel(time='1980-01-01', lat=slice(39.0, 6.2), lon=slice(67, 97.5))  # Replace 'air' variable with temperature in Celsius

# Plotting
fig, ax = plt.subplots()
new_da1.air.plot(ax=ax,cmap='coolwarm', vmin= -20, vmax = 22) # for green and blue plot
#new_da1.air.plot(ax=ax)
# Read and plot the shapefile
shp = gpd.read_file("./india updated state boundary/india updated state boundary.shp")
shp.plot(ax=ax, alpha=0.8, facecolor='None', lw=1)
# Add text description
description = 'Daily Forecast of Air Temperature (°C) at 2m'
plt.text(0.5, 1.09, description, transform=ax.transAxes, ha='center', fontsize=12)


plt.savefig("temp_plot_centigrade3.png")

