# Overview
This repository hosts the implementation for the paper **"Dynamics-enhanced Molecular Property Prediction Guided by Deep Learning"**.

## The package includes:

- Scripts for data preprocessing, feature extraction, model training, and evaluation;
- Complete experimental configurations;
- Implementations of the three DEMR sampling strategies introduced in the paper;
- Instructions for preparing RMSD values for 'Hybrid-category sampling' and 'RMSD-based sampling';
- Links to the MD datasets generated in our study.

------ 

## **Dataset Access**

The molecular dynamics (MD) datasets generated for MPP tasks are published on Zenodo at:

👉 https://doi.org/10.5281/zenodo.15788151[README.md]

Please download the dataset and place it in your corresponding directory. At the same time, modify the data path in project accordingly.

## Python Environment

- Python version: 3.8
- PyTorch: 1.8.1
- CUDA: 10.1

## Feature Extraction

[**database.py**](database.py)

This module extracts the Dynamics-enhanced Molecular Representation (DEMR) proporsed in our study from molecular trajectory files.

Specifically, the script:

- Parses MD trajectory files ('pdb' files);

- Converts raw atomic coordinates into  **dynamics-enhanced molecular representation(DEMR)**;

- Saves DEMR tensors for downstream model training;

Please refer to the usage instructions in the script and place trajectory files according to your directory structure.

## DEMR Sampling Strategies

Our paper introduces three DEMR sampling strategies, each with a corresponding model design:

1. Time-interval Sampling

2. RMSD-based Sampling

3. Hybrid-categorysampling.

## RMSD Requirement



For **Hybrid-category Sampling** and **RMSD-based Sampling**, precomputing the **RMSD** for each frame in the trajectory is required.

- RMSD serves as an additional input to the DataLoader

- It helps identify structurally distinct or representative frames

RMSD values can be computed using the script [**RMSD_analysis.py**](RMSD_analysis.py) included in this package.

For details, see our paper -- **Dynamics-enhanced Molecular Property Prediction Guided by Deep Learning**



## Reproducing Experiments

To reproduce our experimental results:

1. Download the MD dataset from Zenodo

2. Extract DEMR features using database.py

3. Analysis RMSD of the MD trajectory using RMSD_analysis.py

4. Organize dataset files following the structure described in the paper

5. Choose one of the three DEMR sampling strategies

6. Run the training script 



## Citation

If you use this code or dataset, please cite:

"Liu Q, Wang D D, Guo W, et al. Dynamics-enhanced Molecular Property Prediction Guided by Deep Learning"