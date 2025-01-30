# Import packages
import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
# from sklearn.metrics import plot_precision_recall_curve, plot_confusion_matrix, plot_roc_curve
import joblib

"""
Helper functions for preprocessing
"""

def levenshtein_RBD(t, RBD):
    '''
    From Wikipedia article; Iterative with two matrix rows.
    '''
    if RBD == t:
        return 0
    elif len(RBD) == 0:
        return len(t)
    elif len(t) == 0:
        return len(RBD)
    v0 = [None] * (len(t) + 1)
    v1 = [None] * (len(t) + 1)
    for i in range(len(v0)):
        v0[i] = i
    for i in range(len(RBD)):
        v1[0] = i + 1
        for j in range(len(t)):
            cost = 0 if RBD[i] == t[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        for j in range(len(v0)):
            v0[j] = v1[j]

    return v1[len(t)]

"""
Data Encoding Functions
"""

def encode_onehot_padded(aa_seqs):
    '''
    one-hot encoding of a list of amino acid sequences with padding
    parameters:
        - aa_seqs : list with CDR3 sequences
    returns:
        - enc_aa_seq : list of np.ndarrays containing padded, encoded amino acid sequences
    '''
    ### Create an Amino Acid Dictionary
    aa_list = sorted(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
                      'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-'])

    aa_dict = {char: l for char, l in zip(aa_list, np.eye(len(aa_list), k=0))}

    #####pad the longer sequences with '-' sign
    # 1) identify the max length
    max_seq_len = max([len(x) for x in aa_seqs])
    # 2) pad the shorter sequences with '-'
    aa_seqs = [seq + (max_seq_len - len(seq)) * '-'
               for i, seq in enumerate(aa_seqs)]

    # encode sequences:
    sequences = []
    for seq in aa_seqs:
        e_seq = np.zeros((len(seq), len(aa_list)))
        count = 0
        for aa in seq:
            if aa in aa_list:
                e_seq[count] = aa_dict[aa]
                count += 1
            else:
                print("Unknown amino acid in peptides: " + aa + ", encoding aborted!\n")
        sequences.append(e_seq)
    enc_aa_seq = np.asarray(sequences)
    return enc_aa_seq


def flatten_matrix(encoded_aa):
    '''
    simple function to flatten 3D matrix of input AA list of dimensions
    (data_size, positions, one-hot embedding) into
    (data_size, flattened_embedding)
    '''
    mat = encoded_aa
    flat = np.reshape(mat, (mat.shape[0], -1))  # -1 infers last dimension
    return flat




