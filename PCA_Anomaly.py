
"""
This script is for anomaly detection using PCA.
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
data = pd.read_csv('3452_building_data_full_year_copy.csv')

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
_ = plt.title("Importance of the Principal Components based on inertia")
plt.show()

# Calculate PCA with 2 components
scaler = StandardScaler()
pca = PCA(n_components = 2)
pipeline = make_pipeline(scaler, pca)
principalComponents = pipeline.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2'])

# Reconstruct from the 2 dimensional scores
reconstruct = pipeline.inverse_transform(principalComponents)
#The residual is the amount not explained by the first two components
residual = df_tempclean - reconstruct
df_tempclean['pc1'] = pd.Series(principalDf['pc1'].values, index = df_tempclean.index)
df_tempclean['pc2'] = pd.Series(principalDf['pc2'].values, index = df_tempclean.index)

# To detecting the outliers in a data set we have follows the below steps:
# 1 Calculate the 1st and 3rd quartiles.
# 2 Calculates the interquartile range.
# 3 Calculate the upper and lower bound of our data range.
# 4 Using the upper and lower bounds to identify the outlying data points.
# Calculate outlier bounds for pc1
q1_pc1, q3_pc1 = df_tempclean['pc1'].quantile([0.25, 0.75])
iqr_pc1 = q3_pc1 - q1_pc1
lower_pc1 = q1_pc1 - (1.5*iqr_pc1)
upper_pc1 = q3_pc1 + (1.5*iqr_pc1)
# Calculate outlier bounds for pc2
q1_pc2, q3_pc2 = df_tempclean['pc2'].quantile([0.25, 0.75])
iqr_pc2 = q3_pc2 - q1_pc2
lower_pc2 = q1_pc2 - (1.5*iqr_pc2)
upper_pc2 = q3_pc2 + (1.5*iqr_pc2)

# Identification of anomally in original data
df_tempclean['anomaly_pc1'] = ((df_tempclean['pc1'] > upper_pc1) | (df_tempclean['pc1'] < lower_pc1)).astype('int')
df_tempclean['anomaly_pc1'].value_counts()
# Identification of anomally in residuals data
residual['anomaly_pc1'] = ((df_tempclean['pc1'] > upper_pc1) | (df_tempclean['pc1'] < lower_pc1)).astype('int')
residual['anomaly_pc2'] = ((df_tempclean['pc2'] > upper_pc2) | (df_tempclean['pc2'] < lower_pc2)).astype('int')

# Let's plot the outliers from pc1 on top of the device_1783 see where they occured in the time series
for name in names:
    a = df_tempclean[df_tempclean['anomaly_pc1'] == 1] #anomaly
    b = residual[residual['anomaly_pc1'] == 1] #anomaly
    _ = plt.figure(figsize = (18,8))
    _ = plt.plot(df_tempclean['(\'Temperature_01\', 1392)'], color='blue', label='Normal')
    _ = plt.plot(df_tempclean[name], color = 'blue', label = 'Normal')
    _ = plt.plot(residual['(\'Temperature_01\', 1392)'], color = 'black', label = 'Residual')
    _ = plt.plot(residual[name], color = 'black', label = 'Residual')
    _ = plt.plot(a[name], linestyle = 'none', marker = '.', color = 'red', markersize = 12, label = 'Anomaly')
    _ = plt.plot(b[name], linestyle = 'none', marker = '.', color='red', markersize = 12)
    _ = plt.xlabel('Date and Time')
    _ = plt.ylabel('device Reading')
    _ = plt.title('Temperature_01 1392 Anomalies')
    _ = plt.title(name)
    _ = plt.legend(loc = 'best')
    plt.grid()
    plt.autoscale()
    plt.show()
  
