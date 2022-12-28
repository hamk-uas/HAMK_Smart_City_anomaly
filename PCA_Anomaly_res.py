
"""
This script is for anomaly detection using PCA, based on residuals.
To run this script, need to keep data and the script in the same folder.

Input:    
    data - Time series data without N/A and null value

Output:
    plot - Histogram for clean data 
         - Sensor data and residual plot with Anomaly
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
cwd = os.getcwd()


# Read the data set from the folder
data = pd.read_csv('building_data_full_year_copy.csv')


# Convert the data type of timestamp column to datatime format
data['date'] = pd.to_datetime(data['timestamp'])
#del data['timestamp']
# Select only the temperature data and make it index for OT data
df_temp = data.iloc[:, 134:] # 304 for 2948
df_temp = df_temp.set_index('date')
# Remove the zero value and missing value
df_tempclean = df_temp[df_temp != 0]
df_tempclean = df_tempclean.dropna()
# Vizualize time series in the graph for each device
his = df_tempclean.hist(df_tempclean.columns, bins = 25, layout = (8, 8), figsize = (18, 18))
plt.show()
#plt.savefig(his)

# Extract the names of the numerical columns
names = df_tempclean.columns
x = df_tempclean
# Standardize/scale the dataset and apply PCA
scaler = StandardScaler()
pca = PCA()
pipeline = make_pipeline(scaler, pca)
pipeline.fit(x)

features = range(pca.n_components_)
_ = plt.figure(figsize=(22, 5))
_ = plt.bar(features, pca.explained_variance_)
_ = plt.xlabel('PCA feature')
_ = plt.ylabel('Variance')
_ = plt.xticks(features)
_ = plt.title("Importance of the Principal Components based on variance explained")
plt.show()

# Calculate PCA with n components
scaler = StandardScaler()
pca = PCA(n_components = 3)
pipeline = make_pipeline(scaler, pca)
principalComponents = pipeline.fit_transform(x) # Note: this could be done as separate fit and transform steps, using non-anomalous data for fit

# Reconstruct from the n dimensional scores
reconstruct = pipeline.inverse_transform(principalComponents)
#The residual is the amount not explained by the first n components
scaled_residual = pd.DataFrame(data = scaler.transform(x) - pca.inverse_transform(principalComponents), index = df_tempclean.index, columns=df_tempclean.columns)
residual = df_tempclean - reconstruct

# Detect outliers based on scaled residual variance summed over principal component features
scaled_residual_total_var_anomaly_threshold = 5 # Arbitrary threshold
scaled_residual_sq_anomaly_threshold = 5 # Arbitrary threshold
scaled_residual_var = (scaled_residual ** 2).sum(axis = 1)
is_anomaly_based_on_scaled_residual_total_var = scaled_residual_var > scaled_residual_total_var_anomaly_threshold
is_anomaly_based_on_scaled_residual = (scaled_residual ** 2) > scaled_residual_sq_anomaly_threshold

# Let's plot the outliers from pc1 on top of the device_1783 see where they occured in the time series
a1 = df_tempclean[is_anomaly_based_on_scaled_residual_total_var] #anomaly
b1 = residual[is_anomaly_based_on_scaled_residual_total_var] #anomaly
a2 = df_tempclean[is_anomaly_based_on_scaled_residual] #anomaly
b2 = residual[is_anomaly_based_on_scaled_residual] #anomaly
for name in names:
    _ = plt.figure(figsize = (18,8))
    #_ = plt.plot(df_tempclean['(\'Temperature_01\', 1392)'], color='blue', label='Normal')
    _ = plt.plot(df_tempclean[name], color = 'blue', label = 'Feature')
    #_ = plt.plot(residual['(\'Temperature_01\', 1392)'], color = 'black', label = 'Residual')
    _ = plt.plot(residual[name], color = 'black', label = 'Residual')
    _ = plt.plot(a1[name], linestyle = 'none', marker = '.', color = 'orange', markersize = 12, label = 'Anomaly based on all residuals')
    _ = plt.plot(a2[name], linestyle = 'none', marker = '.', color = 'red', markersize = 8, label = 'Anomaly based on residual')
    _ = plt.plot(b1[name], linestyle = 'none', marker = '.', color='orange', markersize = 12)
    _ = plt.plot(b2[name], linestyle = 'none', marker = '.', color='red', markersize = 8)
    _ = plt.xlabel('Date and Time')
    _ = plt.ylabel('device Reading')
    #_ = plt.title('Temperature_01 1392 Anomalies')
    _ = plt.title(name)
    _ = plt.legend(loc = 'best')
    plt.grid()
    plt.autoscale()
    plt.show()

