![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
# Adversarial Machine Learning Mitigation
### 

## Author: 
- Fridtjof Storm Flaate | fridtjsf@stud.ntnu.no

## Table of Contents

- [Project Description](#project-description)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)

## Project Description

In this project we will implement two CNN's, Inception V3 and Resnet-18. These models will be trained on the ISIC2018 and ISIC2019 datasets. Furthermore, we will implement two types of adversarial attacks to pertubate input images and try to fool the models. When this milestone is finished, we will extract kernels and other parameters from the model to train a adversarial detector model, which are trained on perturbated and non-pertubated images.

## Project Structure

- `data/` - Directory for storing data files
- `augmented_data/` - Directory for storing augmented data files
- `models/` - Default directory for trained and tested models
- `run_*` - Files used to run python scripts
   - [`run_test_multiprocessing.py`](./run_test_multiprocessing.py): Script to evaluate the number of processors your computer can handle
   - [`run_data_augmentation.py`](./run_data_augmentation.py): Script to perform the data augmentation of dataset. Saves the augmented data into the [`augmented_data/`](./augmented_data/) directory.
   - [`run_train_model.py`](./run_train_model.py): Script to load in data, train and test model, uses parameters defined in the [`config.py`](./config.py) file. Will also save the model and the config.py file used to get that model.
- `*_helper` - Files used for helper functions
   - [`train_model_helper.py`](./train_model_helper.py): Helper functions related to the [`run_train_model.py`](./run_train_model.py) file.
      - `train_model()`
      - `test_model()`
      - `get_category_counts()`
      - `random_split()`
      - `get_data_loaders()`
   - [`data_exploration_helper.py`](./data_exploration_helper.py): Helper functions related to EDA (Exploratory Data Analysis) 
      - `dataset_overview()`
      - `perform_eda()`
   - [`misc_helper.py`](./misc_helper.py): A collection of miscellaneous helper functions that do not have a specific category assignment
      - `truncated_uuid4()`
      - `get_trained_or_default_model()`
      - `save_model_and_parameters_to_file()`
      - `load_model_from_file()`
      - `file_exists()`
      - `folder_exists()`
   - [`adversarial_attacks_helper.py`](./adversarial_attacks_helper.py): Helper functions related to adversarial attacks
- [`customDataset.py`](./customDataset.py): Custom dataset which inherits from `torch.data.utils.Dataset`. Collects dataset based on the value provided in [`config.py`](./config.py) parameter file.
- `README.md` - This file, containing information about the project.

## Getting Started

1. Clone the repo and enter directory `adversarial_mitigation`
   ```sh
   git clone https://github.com/StormFlaate/adversarial_mitigation.git
   ```
2. create environment
   ```sh
   conda create -n adv_mit python=3.10
   ```
3. activate environment
   ```sh
   conda activate adv_mit
   ```