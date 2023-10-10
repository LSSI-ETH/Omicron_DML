# Omicron_DML
Code repo associated with the Frei_Gao_2023 manuscript

Packages in requirements.txt requires Python version 3.9.0

=====================================================

The following scripts are part of the DML pipeline as described in the paper:

base_models.py: Trains a set of baseline models available from the scikit-learn package on labelled datasets, performs hyper parameter search through RandomSearchCV, and stores the scores of the best performing models for comparison with deep learning models.

DL_models.py: Trains either a multilayer perceptron (‘mlp’) or convolutional neural network (‘cnn’) based on given arguments, with class balancing, as well as hyper parameter optimisation if desired. Saves final model, its scores on the test set, and the predictions on a curated set of major SARS-CoV-2 variant RBD sequences. 

Top_models.py: Analyses all deep learning models trained so far, and finds the top N models to be used in ensemble predictions.

Lineage_generator.py: Uses given GISAID sequences to generate a probability matrix, then generates novel variants based on a given starting seed sequence. Requires ACE2 models to have been trained from DL_models.py and Top_models.py.

Breadth_predictions.py: Uses generated sequences from Lineage_generator.py, as well as the final models selected by Top_models.py for breadth predictions of specified mAbs. 

=====================================================

./data only contains a small subset of the data from our sorted libraries. Please feel free to reach out to the authors to request the full datasets.
