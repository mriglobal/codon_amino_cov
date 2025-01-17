# Code and Data For Coronavirus Human-Pathogen Models

# Repo organization

 - /data
	 - supplementary_data1.zip contains all of the results summary, persistent model objects, and other data summaries used in the paper
	 - seq_data.zip contains all of the sequence data and label metadata used in the modeling
 - /routines
	 - Contains the scripts used to fit the RSCU, Codon, and Amino Acid Frequencies models
 - /utils
	 - Contains the cross validation and feature extraction classes needed to fit the models and inference on new data
	 - Jupyter notebook to easily load models and get inference on new data
 - /configuration_files
	 - Example YAML files used in conjunction with model scripts

environment.yaml - Environment YAML file needed to recreate the python environment.

## Environment Setup

To install the dependencies to run the models after cloning this repo:


    conda env create -f environment.yml

Activate the environment:

    conda activate codon_amino_cov

## Load model and test on new data

Unzip the data located in /data

    unzip codon_amino_cov.zip

While in the repository directory start up a Jupyter notebook:

    jupyter notebook --port=8883

Replace the empty path strings with the locations of the /utils directory for this repository, the models of interest and the data you want to inference on and run the notebook cells.

