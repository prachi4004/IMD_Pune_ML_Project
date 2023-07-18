import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import xarray as xr
import numpy as np
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dense, Dropout, ELU
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

############################################### Data Pre-Processing ###########################################################
df=pd.read_csv('Panjim_Data.csv')
year=df['Year']
#print(year)
rf=[]
for i in range(0,year.shape[0]):
    if(year[i]%4==0):
        for j in range(1, 367):             #loop over all columns named 1 to 366
            rf.append(df[str(j)][i])
    else:
        rf_tmp=[]
        for j in range(1, 367):
            if(j!=60):
                rf_tmp.append(df[str(j)][i])
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

RF_final_time= xr.Dataset(data_vars={'rfl': RF_final}, coords={'time': Time})
#print(RF_final_time)

########################################################## Model ################################################################################
#Splitting thedata into input X and output y
def split_sequence(sequence, n_steps_in, n_steps_out):
 X, y = list(), list()
 for i in range(len(sequence)):
     # find the end of this pattern
     end_ix = i + n_steps_in
     out_end_ix = end_ix + n_steps_out
     # check if we are beyond the sequence
     if out_end_ix > len(sequence):
         break
     # gather input and output parts of the pattern
     seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
     X.append(seq_x)
     y.append(seq_y)
 return array(X), array(y)

# choose a number of time steps
n_steps_in, n_steps_out = 7, 1         #the model will take input as rainfall data of past 7 days and give the rainfall prediction for the next 1 day

# split into samples
X, y = split_sequence(RF_final, n_steps_in, n_steps_out)
print(X.shape)
print(y.shape)

# Split dataset into training set and test set, 77.990% is kept as training set and 22.009 % is kept as testing set
X_train_l,X_test_l=list(), list()
y_train_l,y_test_l=list(), list()

for i in range(len(X)):                    
     if(i<=14237):                #14238 is the day number of 2010-01-01 which is the starting point of testing set (day 1 would be 1971-01-01,day 366 would be 1972-01-01,day 731 would be 1973-0-01 and so on)
         X_train_l.append(X[i])
         y_train_l.append(y[i])
     else:
         X_test_l.append(X[i])
         y_test_l.append(y[i])


X_train=array(X_train_l)
X_test=array(X_test_l)
y_test=array(y_test_l)
y_train=array(y_train_l)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from keras.layers import ELU
# define model
model = Sequential()
model.add(Dense(14, activation=ELU(alpha=1.0), input_shape=(n_steps_in,)))
model.add(Dense(10, activation=ELU(alpha=1.0)))
model.add(Dense(12, activation=ELU(alpha=1.0)))
model.add(Dense(n_steps_out, activation=ELU(alpha=1.0)))
model.compile(optimizer='adam', loss='mse')



# Fit the model and store the history
history = model.fit(X_train, y_train, epochs=200, batch_size=64, verbose=1, validation_data=(X_test, y_test))




#Plot the training loss and test loss
train_loss = history.history['loss']
test_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)


plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, test_loss, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_Vs_Epochs_mlp_model.png')

predictions = model.predict(X_test)

# Calculate RMSE value on the testing set
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("RMSE on testing set:", rmse)

# Calculate the correlation coefficient
corr_coefficient, _ = pearsonr(y_test.flatten(), predictions.flatten())
print("Correlation coefficient:", corr_coefficient)

# Calculate Standardized RMSE value on the testing set
std_deviation=np.std(RF_final)                #Calculating the standard deviation
#print(std_deviation)
print(f"Standardised RMSE: {rmse/std_deviation}")


# Save the model and its weights
model.save('Panjim_mlp_model.h5')
