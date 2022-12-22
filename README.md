# HAMK Smart City project - Anomaly detection in HVAC building data.
This is a repository include Python3 scripts and Python Notebook for anomaly detection in Smart City project. The Smart City project of Häme Universty of Applied Sciences focuses on promoting low-carbon developments through imlementation of Artificial Intelligence. The content of this repository is focused on anomaly detection on heating energy consumption of HVAC system control.

## Introduction

Principal Component Analysis (PCA) is a method for dimensionality-reduction by transforming (Singular Value Decomposition of the data) a large set of variables into a smaller one that still contains most of the information in the large set. The input data is centered but not scaled for each feature before applying the SVD.

## Installation

Install required python libraries by using requirements.txt

>pip install -r requirements.txt


## Code run

Need to keep data and file in the same folder to run the Python scripts.

## Overview of files

* PCA_Anomaly.py - Python script for anomaly detection using PCA and Quartile based method.

* PCA_Anomaly.ipynb - Notebook for identification of anomaly using PCA and Quartile based method. Step-by-step explanation and visualization have performed.

* PCA_Anomaly_res.py - Python script for identification of anomaly using PCA and residual based method.

* PCA_Anomaly_res_tuning.py - Fine tuned code for PCA and residual based method for the identification of anomaly using best components.

* PCA_Anomaly_res_tuning_work.py - PCA and residual based method for anomaly detection trained on clean data and testing on manually contaminated data.

## Recomendations

* For this kind of specific data, PCA based method is most suitable although depending on the questions that need to be answered, need to be more careful to choose the unsupervised method.
* For higher dimentional data, it is better to reduce the dimentionality for better outcomes before selecting anomaly detection rules.
* It will be always good choice to use supervised method, if there exist labels.

## Author

2022@ Abdur Rahman

## Licence

Permissive Apache License 2.0
