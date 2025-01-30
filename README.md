# Omicron_DML
Code repo associated with the Frei_Gao_2023 manuscript

Packages in requirements.txt requires Python version 3.9.0

=====================================================

The folders within the repo contain scripts that are part of the DML pipeline as described in the paper:

models: contains scripts that trains a small set of baseline models, or deep learning models (MLP and CNN) used in the paper. With train_dl_models.py, multiple models can be trained with with class balancing, as well as hyper parameter optimisation if desired. Saves final model, its scores on the test set, and the predictions on a curated set of major SARS-CoV-2 variant RBD sequences.  

breadth_prediction: contains scripts that analyses and selects the top N deep learning models to be used in ensemble predictions, then uses GISAID probabilities, and ACE2 models to generate novel lineages of variants based on a given starting seed sequence. Finally, uses mAb models to predict the breadth of specified mAbs. 

escape_score: contains R scripts used to calculate escape score of synthetic lineages and variants based on predicted mAb breadth.

utils: contains utility functions used throughout

=====================================================

NOTE: ./data only contains a small subset of the data from our sorted libraries. Please feel free to reach out to the authors to request the full datasets.
