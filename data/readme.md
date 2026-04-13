# Dataset Instructions

This project uses the UNSW-NB15 dataset for anomaly-based intrusion detection.

## Download Dataset

Download the dataset from the official source:
https://research.unsw.edu.au/projects/unsw-nb15-dataset

## Required Files

After downloading, place the following files in this directory:

* UNSW_NB15_training-set.csv
* UNSW_NB15_testing-set.csv

Optional:

* UNSW-NB15_features.csv (for feature reference)

## Folder Structure

data/
│── UNSW_NB15_training-set.csv
│── UNSW_NB15_testing-set.csv
│── UNSW-NB15_features.csv

## Notes

* The dataset is not included in this repository due to size constraints.
* The model is trained only on normal traffic (label = 0) to simulate real-world anomaly detection.
* Testing is performed on both normal and attack traffic.

## Preprocessing

The dataset will be automatically preprocessed during model training:

* Missing value handling
* Encoding categorical features
* Feature scaling

No manual preprocessing is required before running the project.
