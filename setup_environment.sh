#!/bin/bash

# Create conda environment from yml
conda env create -f environment.yml

# Activate the environment
conda activate your_env_name

# Install pip requirements
pip install -r requirements.txt
