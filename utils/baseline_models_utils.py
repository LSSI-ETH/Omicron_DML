"""
Author: Andrey Ignatov
Email: andrey.ignatoff@gmail.com

This script implements auxiliary data processing functions required for baseline models training.
"""

import os.path

import numpy as np
import pandas as pd

from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

np.random.randint(42)


def filter_dataset(data, labels, counts, threshold_count):

    # Filter sequences with counts less than threshold_count

    filtered_idx = np.argwhere(counts >= threshold_count)[:, 0]

    data = data[filtered_idx]
    labels = labels[filtered_idx]

    return data, labels


def load_data(target, lib, path_to_dataset):

    # Load sequences, labels and sequence counts

    dataset_path = os.path.join(path_to_dataset, lib, target + "_" + lib + "_labeled.csv")

    data = pd.read_csv(dataset_path)

    sequences = np.asarray(data["aa"])
    seq_count = np.asarray(data["Total_sum"])
    seq_label = np.asarray(data["Label"])

    return sequences, seq_label, seq_count


def encode_onehot_padded(aa_seqs, flatten=False):

    # Perform one hot encoding of the sequence data

    aa_list = sorted(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])
    aa_dict = {char: l for char, l in zip(aa_list, np.eye(len(aa_list), k=0))}

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

        if flatten:
            sequences.append(np.ndarray.flatten(e_seq))
        else:
            sequences.append(e_seq)

    enc_aa_seq = np.asarray(sequences)
    return enc_aa_seq


def encode_numerical(aa_seqs):

    # Encode sequence data as numerical features

    sequences = []
    for seq in aa_seqs:

        data_raw = list(seq)
        data = []
        for aa in data_raw:
            data.append(ord(aa) - 65)

        sequences.append(data)

    enc_aa_seq = np.asarray(sequences)
    return enc_aa_seq


def compute_accuracy_scores(targets, predictions):

    # Compute accuracy, precision, recall, F1 and MCC scores

    accuracy = accuracy_score(targets, predictions) * 100
    precision = precision_score(targets, predictions) * 100
    recall = recall_score(targets, predictions) * 100
    f1 = f1_score(targets, predictions)
    mcc_score = mcc(targets, predictions)

    return [accuracy, precision, recall, f1, mcc_score]
