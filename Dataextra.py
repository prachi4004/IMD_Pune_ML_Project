
##########################Import necessary Modules###################
import xarray as xr

##########Read the dataset using xarray##########################
filename=r"C:\Users\Prachi\OneDrive\Documents\NCEP_Reanalysis2_data\2m_Temp\air.2m.gauss.1980.nc"
da=xr.open_dataset(filename)
print(da)

############Read the individual variables in the dataset########
air=da.air
lat=da.lat
lon=da.lon
print(lon)