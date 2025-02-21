# Code and Data For Coronavirus Human-Pathogen Models

# Repo organization

 -Model data, sequence data, and modeling summary files are available at https://doi.org/10.5281/zenodo.14851561. The repository contains a README file that explains the organization of the Supplementary Data.
 - /routines
	 - Contains the scripts used to fit the RSCU, Codon, and Amino Acid Frequencies models
 - /utils
	 - Contains the cross validation and feature extraction classes needed to fit the models and inference on new data
	 - Jupyter notebook to easily load models and get inference on new data
 - /configuration_files
	 - Example YAML files used in conjunction with model scripts

requirements.txt - A concise list of the dependencies needed to recreate the modeling environment

## Environment Setup

To install the dependencies to run the models after cloning this repo:


    conda create --name codon_amino_cov --file requirements.tx

Activate the environment:

    conda activate codon_amino_cov

## Load model and test on new data

Download the data from the Zenodo repository and unzip in a local directory

    unzip codon_amino_rscu_final.zip

While in the repository directory start up a Jupyter notebook:

    jupyter notebook --port=8883

Replace the empty path strings in the sys.path with the locations of the /utils directory for this repository. Instructions on how to load the models of interest and the data you want to inference on are located in the comments of the notebook cells.
