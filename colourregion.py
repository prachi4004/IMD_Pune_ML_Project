import xarray as xr
import matplotlib.pyplot as plt
#import geopandas as gpd

##########Read the dataset using xarray##########################
filename="./NCEP_Reanalysis2_data/2m_Temp/air.2m.gauss.1980.nc"
da=xr.open_dataset(filename)
print(da)

############Read the individual variables in the dataset########
air=da.air
lat=da.lat
lon=da.lon
print(lon)


################ Subsetting in all dimensions###################
da1=da.sel(time='1980-02-02',lat=slice(40.5,7.5),lon=slice(68.0,90.5))
print(da1)
print(da1.air)

plot=da1.air.plot()

plt.savefig("testcase3.png")