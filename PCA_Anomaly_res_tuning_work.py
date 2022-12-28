
"""
This script is for anomaly detection using PCA, based on residuals.
To run this script, need to keep data and the script in the same folder.

Input:    
    data - Time series data without N/A and null value
         - Contaminated data with manual anomaly
         - Training data remains anomaly free

Output:
    plot - Histogram for clean data 
         - Sensor data and residual plot with Anomaly
         - Colour bar plot for total variance and individual variance
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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
#df_tempclean = df_tempclean[(df_tempclean.index >= "2021-12-01") & (df_tempclean.index < "2022-03-16")]
# Training data
df_tempclean_train = df_tempclean[(df_tempclean.index >= "2021-12-01") & (df_tempclean.index < "2022-02-16")]
# Test data
df_tempclean_test = df_tempclean[(df_tempclean.index >= "2022-02-17") & (df_tempclean.index < "2022-03-16")]

# Create anomalous data
anomalous_len = int((len(df_tempclean_test) - 0.30*len(df_tempclean_test)) / 2)  # 30 % of anomalous window to mess up
df_anomalous = df_tempclean_test.tail(anomalous_len + anomalous_len)  # Get the tail end of our raw data
df_anomalous = df_anomalous.head(anomalous_len)  # Take the top part of it
df_anomalous = df_anomalous.rolling(100).mean()*2  # Apply a rolling avg to smooth it

# Append test data and anomalous data as 'contaminated' data
df_contaminated = df_tempclean_test.append(df_anomalous).append(df_tempclean_test.tail(anomalous_len)).interpolate(method = 'linear')
df_contaminated.plot(title = 'contaminated data')
#his = df_contaminated.hist(df_contaminated.columns, bins = 25, layout = (8, 8), figsize = (18, 18))
plt.show()


#df_tempclean_train, df_tempclean_test, label_train, label_test = train_test_split(df_tempclean, label, test_size = 0.2)

# Extract the names of the numerical columns
names = df_tempclean_train.columns
x = df_tempclean_train
# Standardize/scale the dataset and apply PCA

# To see the importance of PC's, need to uncomment below code
#features = range(pca.n_components_)
#_ = plt.figure(figsize=(22, 5))
#_ = plt.bar(features, pca.explained_variance_)
#_ = plt.xlabel('PCA feature')
#_ = plt.ylabel('Variance')
#_ = plt.xticks(features)
#_ = plt.title("Importance of the Principal Components based on variance explained")
#plt.show()

n_components_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
threshold_list = np.arange(0.125, 40, 0.125)
result_table_total_var = np.zeros((len(n_components_list), len(threshold_list)))
result_table_individual = np.zeros((len(n_components_list), len(threshold_list)))
for n_components_index, n_components in enumerate(n_components_list):
    scaler = StandardScaler()
    pca = PCA(n_components = n_components)
    pipeline = make_pipeline(scaler, pca)
    principalComponents_train = pipeline.fit_transform(x) # Note: this could be done as separate fit and transform steps, using non-anomalous data for fit
    
    # Reconstruct from the n dimensional scores
    reconstruct = pipeline.inverse_transform(principalComponents_train)
    #The residual is the amount not explained by the first n components
    scaled_residual = pd.DataFrame(data = scaler.transform(x) - pca.inverse_transform(principalComponents_train), index = df_tempclean_train.index, columns = df_tempclean_train.columns)

    for threshold_index, threshold in enumerate(threshold_list):
        scaled_residual_var = (scaled_residual ** 2).sum(axis = 1)
        is_anomaly_based_on_scaled_residual_total_var = scaled_residual_var > threshold
        is_anomaly_based_on_scaled_residual = (scaled_residual ** 2) > threshold
        result_table_total_var[n_components_index, threshold_index] = is_anomaly_based_on_scaled_residual_total_var.astype('int').sum().sum()
        result_table_individual[n_components_index, threshold_index] = is_anomaly_based_on_scaled_residual.astype('int').sum().sum()


#print(result_table_total_var)
#print(result_table_individual)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.title('Total Variance for Train')
plt.imshow(result_table_total_var, interpolation = 'nearest', vmin = 0, vmax = 10, aspect = (threshold_list[-1] - threshold_list[0])/(n_components_list[-1] - n_components_list[0]), extent = [threshold_list[0], threshold_list[-1], n_components_list[0]-0.5, n_components_list[-1]+0.5], origin = 'lower') # , cmap = plt.cm.ocean
plt.colorbar()
plt.show()
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.title('Individual Variance for Train')
plt.imshow(result_table_individual, interpolation = 'nearest', vmin = 0, vmax = 10, aspect = (threshold_list[-1] - threshold_list[0])/(n_components_list[-1] - n_components_list[0]), extent=[threshold_list[0], threshold_list[-1], n_components_list[0]-0.5, n_components_list[-1]+0.5], origin = 'lower') # , cmap = plt.cm.ocean
plt.colorbar()
plt.show()


for n_components_index, n_components in enumerate(n_components_list):
    scaler = StandardScaler()
    pca = PCA(n_components = n_components)
    pipeline = make_pipeline(scaler, pca)
    principalComponents_train = pipeline.fit_transform(x)
    principalComponents_contaminated = pipeline.transform(df_contaminated)
    # Reconstruct from the n dimensional scores
    reconstruct = pipeline.inverse_transform(principalComponents_contaminated)
    #The residual is the amount not explained by the first n components
    scaled_residual = pd.DataFrame(data = scaler.transform(df_contaminated) - pca.inverse_transform(principalComponents_contaminated), index = df_contaminated.index, columns = df_contaminated.columns)

    for threshold_index, threshold in enumerate(threshold_list):
        scaled_residual_var = (scaled_residual ** 2).sum(axis = 1)
        is_anomaly_based_on_scaled_residual_total_var = scaled_residual_var > threshold
        is_anomaly_based_on_scaled_residual = (scaled_residual ** 2) > threshold
        result_table_total_var[n_components_index, threshold_index] = is_anomaly_based_on_scaled_residual_total_var.astype('int').sum().sum()
        result_table_individual[n_components_index, threshold_index] = is_anomaly_based_on_scaled_residual.astype('int').sum().sum()

#print(result_table_total_var)
#print(result_table_individual)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.title('Total Variance for Contaminated')
plt.imshow(result_table_total_var, interpolation = 'nearest', vmin = 0, vmax = 10, aspect = (threshold_list[-1] - threshold_list[0])/(n_components_list[-1] - n_components_list[0]), extent = [threshold_list[0], threshold_list[-1], n_components_list[0]-0.5, n_components_list[-1]+0.5], origin = 'lower') # , cmap = plt.cm.ocean
plt.colorbar()
plt.show()
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.title('Individual Variance for Contaminated')
plt.imshow(result_table_individual, interpolation = 'nearest', vmin = 0, vmax = 10, aspect = (threshold_list[-1] - threshold_list[0])/(n_components_list[-1] - n_components_list[0]), extent=[threshold_list[0], threshold_list[-1], n_components_list[0]-0.5, n_components_list[-1]+0.5], origin = 'lower') # , cmap = plt.cm.ocean
plt.colorbar()
plt.show()