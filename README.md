# HAMK Smart City project - Anomaly detection in HVAC building data.
This repository includes Python3 scripts and a Python Notebook for anomaly detection in the Smart City project. The Smart City project of Häme Universty of Applied Sciences (HAMK) focuses on reducing carbon emissions using Artificial Intelligence. The content of this repository is focused on anomaly detection in HVAC temperature data.

## Introduction

Principal Component Analysis (PCA) is a method for dimensionality reduction by transforming (Singular Value Decomposition of the data) a large set of variables into a smaller set that still contains most of the information in the large set. The input data is centered but not scaled for each feature before applying the SVD. We have used PCA for dimension reduction for the data and after that used both residual based methods and a quartile based method to define the anomaly range.

## Data

The dataset is in 'Data/building_data_full_year_copy.csv'. This is a yearly time series sensor data including temperatures (among other variables) at hourly time resolution.

## Installation

Install required python libraries listed in requirements.txt

>pip install -r requirements.txt

## Running

Keep the data file in the same folder to run the Python scripts.

## Overview of files

* PCA_Anomaly.py - Python script for anomaly detection using PCA and the quartile based method.

* PCA_Anomaly.ipynb - Jupyter Notebook for anomaly detection using PCA and the quartile based method. Includes step-by-step explanation and visualization.

* PCA_Anomaly_res.py - Python script for anomaly detection using PCA and residual based methods.

* PCA_Anomaly_res_tuning.py - Fine tuned code for PCA and residual based method for anomaly detection using best components.

* PCA_Anomaly_res_tuning_work.py - PCA and residual based method for anomaly detection trained on clean data and testing on contaminated data with synthetic anomalies.

## Recommendations

* For the present dataset, a PCA based method is suitable. The best choice of the unsupervised anomaly detection method will depend on the questions to be answered
* For higher-dimensional data, it is better to reduce the dimensionality for better outcomes before selecting anomaly detection rules.
* It will be always good choice to use a supervised method, if labels (anomaly / no anomaly) exist.

## Authors

* Abdur Rahman
* Olli Niemitalo

## Copyright

Copyright 2022-2023, Häme University of Applied Sciences (HAMK), Finland

## Licence

Dual licensed under the permissive Apache License 2.0 and MIT licenses.
