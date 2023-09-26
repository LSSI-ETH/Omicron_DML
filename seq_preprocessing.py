#!/cluster/apps/nss/gcc-8.2.0/python/3.10.4/x86_64/bin/python
# -*- coding: utf-8 -*-

# %%
"""
Authors: Beichen Gao, Benedikt Eisinger, Rachita Kumar
Script for processing of binding and non-binding NGS datasets into the final dataset used for model training

All functions use relative paths to save outputs because the main
function that is called creates a directory structure and switches directories.

Before starting you need to change:
    WORK_DIR, SAVE_DIR, RAW_SEQ_DIR
    WORK_DIR should contain helpers.py

Some functions have inputs that determine e.g. how many sequences are filtered and might need to be adjusted
for a specific purpose.

To run the script you have to:
    - Define a list object with the library numbers e.g. ['lib1', 'lib2', ...]
    - Define a list object with the sequence origins e.g. ['ACE2', 'naive']
    - Define the original RBD sequence for distance calculations

In the grand scope of things, this script is likely unnecessary to understand DML
However, we're keeping it in the repo for people to understand how we preprocessed the data beforehand
"""

import os
import pathlib
import pandas as pd
from helpers import levenshtein_RBD

# set directories if working locally. File path should have a '/' at the end!!!!
WORK_DIR = os.getcwd()
# SAVE_DIR is the location for all output.
SAVE_DIR = f"{WORK_DIR}/data"
# RAW_SEQ_DIR is the location of the raw sequences you want to process
RAW_SEQ_DIR = f"{WORK_DIR}/raw_data"
os.chdir(WORK_DIR)

rand_state = 42

"""
Create required functions
"""

def make_full_seq(df, missing_seq, target, lib_name):
    """
    This function concatenates the wt sequence to the one that contains the mutation. Additionally, it
    removes the first two AAs from the lib1 sequences. This function is specifically made for one type of dataset!
    Inputs:
        df: (df) Dataframe containing aa sequences and counts from NGS sequencing of a sorted population
        missing_seq: (str) sequence to be added
        lib_name: (str) Name of the library which was sorted (ex. "Lib1" for seq-libraryA)
    """
    # To ensure that the original df is not overwritten we need to specifically copy it not a=df is only a reference
    # and does not suffice
    a = df.copy()
    if lib_name == 'Lib1':
        a["aa"] = a["aa"].str[2:]
        print("First two characters have been removed from Lib1 sequences. \n")
        a["aa"] = a["aa"].astype(str) + str(missing_seq)
        print("Lib1 mutated and wt seq have been concatenated. \n")
        a["full_aa_length"] = (a['aa_length'] + len(missing_seq)) - 2
        a['Library'] = 'Lib1'
    elif lib_name == 'Lib2':
        a["aa"] = str(missing_seq) + a["aa"].astype(str)
        print("Lib2 mutated and wt seq have been concatenated. \n")
        a["full_aa_length"] = (a['aa_length'] + len(missing_seq))  # - 2
        a['Library'] = 'Lib2'
    else:
        print("Library not anticipated, this code may need to be updated. Process aborted!")
    a['Target'] = target
    return a


def clean_df(df, target, lib_name, label, wt_seq, threshold, SAVE_DIR):
    """

    Parameters
    ----------
    df: (df) Dataframe containing aa sequences and counts from NGS sequencing of a sorted population
    target: (str) Name of the target used for selections
    lib_name: (str) Name of the library which was sorted (ex. "Lib1" for seq-libraryA)
    label: (str) Label of the population (ex. 'B' for binding population)
    wt_seq: (str) full sequence of main RBD variant (ex. BA.1)
    threshold: (int) minimum count threshold of each sequence for filtering and preprocessing

    Returns
    -------

    """
    # Drop columns from df
    processed_df = df.copy()
    processed_df = processed_df.drop(columns=['aa_length'])

    # aggregate all counts of the same AA sequence
    processed_df['Total_sum'] = processed_df.groupby(['aa'])['n'].transform('sum')
    processed_df = processed_df.drop_duplicates(subset='aa')
    processed_df = processed_df.drop(['n'], axis=1)

    # remove sequences with stop codons and "X"
    processed_df = processed_df[~processed_df.aa.str.contains(r'[*]')]
    processed_df = processed_df[~processed_df.aa.str.contains(r'[X]')]

    # drop all sequences below count threshold
    processed_df = processed_df[processed_df['Total_sum'] >= threshold]

    # keep sequences that are 201 in length only
    processed_df = processed_df[processed_df['full_aa_length'] == 201]

    # reset index
    processed_df = processed_df.reset_index(drop=True)

    # add LD column
    processed_df['Distance'] = processed_df.apply(lambda row: levenshtein_RBD(row.aa, wt_seq), axis=1)

    # Save edited df
    processed_df.to_csv(f'{SAVE_DIR}/{target}_{lib_name}_{label}_cleaned.csv')

    # return df as well
    return processed_df


def create_labeled_df(bind_df, escape_df, target, lib_name, SAVE_DIR):
    """

    Parameters
    ----------
    bind_df: (df) Cleaned dataframe containing binding sequences
    escape_df: (df) Cleaned dataframe containing non-binding/escape sequences
    target: (str) Name of the target used for selections
    lib_name: (str) Name of the library which was sorted (ex. "Lib1" for seq-libraryA)
    SAVE_DIR: (str) Path to save directory for processed files

    Returns
    -------
    Labeled dataframe containing unique sequences in binding and non-binding sorts

    """
    # specify main columns used for future analysis
    main_cols = ['aa', 'Total_sum', 'freq', 'full_aa_length', 'Library', 'Target', 'Distance']
    bind_df = bind_df[main_cols].reset_index(drop=True)
    escape_df = escape_df[main_cols].reset_index(drop=True)

    # add labels
    bind_df['Label'] = 1
    escape_df['Label'] = 0

    # concatenate dfs, then drop the duplicates
    merged_df = pd.concat([bind_df, escape_df])
    processed_df = merged_df.drop_duplicates(subset='aa', keep=False)

    # Save the df with duplicates
    processed_df.to_csv(f'{SAVE_DIR}/{target}_{lib_name}_labeled.csv')

    # return labelled df
    return processed_df


"""
Main function
"""

def main(RAW_SEQ_DIR, SAVE_DIR, lib_list, targets_list, labels_list, wt_seq, common_naming, threshold):
    """

    Parameters
    ----------
    RAW_SEQ_DIR: (str) Path to directory where raw sequences with counts are stored
    SAVE_DIR: (str) Path to save directory, where all processed files are saved to
    lib_list: (list) list of names of libraries to be processed (ex. ['Lib1', 'Lib2']
    targets_list: (list) list of names of targets (ex. 'ACE2' or mAbs)
    labels_list: (list) list of labels (ex. ['B', 'E']
    wt_seq: (str) full sequence of main RBD variant (ex. BA.1)
    common_naming: (str) additional common name added to the end of all FASTA processed .csv files (ex. '_notrim')
    threshold: (int) minimum count threshold of each sequence for filtering and preprocessing

    Returns
    -------

    """
    # iterate through and process all FASTA extracted csv's together
    for lib in lib_list:
        # make specific save dir
        LIB_SAVE_DIR = f'{SAVE_DIR}/{lib}'
        pathlib.Path(LIB_SAVE_DIR).mkdir(parents=True, exist_ok=True)

        # specify missing portion of RBD from each library
        missing = ''
        if lib == 'Lib1':
            missing = "NKPCNGVAGFNCYFPLRSYSFRPTYGVGHQPYRVVVLSFELLHAPATVCGPKKST"
        elif lib == 'Lib2':
            missing = "NITNLCPFDEVFNATRFASVYAWNRKRISNCVADYSVLYNLAPFFTFKCYGVSP"

        # iterate through targets and labels
        for target in targets_list:
            for label in labels_list:
                print(f'Now processing {target}_{lib}_{label} sequences! \n')

                # Import data
                seq_df = pd.read_csv(f'{RAW_SEQ_DIR}/{target}_{lib}_{label}_{common_naming}.csv')

                # Make full sequences
                seq_df_full = make_full_seq(seq_df, missing, target, lib)

                # Clean the sequences
                seqs_cleaned = clean_df(seq_df_full, target, lib, label, wt_seq, threshold, LIB_SAVE_DIR)

            # Read in the cleaned dataframes for binding and escape
            bind_df = pd.read_csv(f'{LIB_SAVE_DIR}/{target}_{lib}_B_cleaned.csv')
            escape_df = pd.read_csv(f'{LIB_SAVE_DIR}/{target}_{lib}_E_cleaned.csv')

            # Make final labelled dataframe
            labeled_df = create_labeled_df(bind_df, escape_df, target, lib, LIB_SAVE_DIR)


# The following are some default options for running the preprocessing
targets_list = ['ACE2', 'ZCB11', 'S2X259']
lib_list = ['Lib1', 'Lib2']
labels_list = ['B', 'E']
BA1_seq = 'NITNLCPFDEVFNATRFASVYAWNRKRISNCVADYSVLYNLAPFFTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQT' \
                 'GNIADYNYKLPDDFTGCVIAWNSNKLDSKVSGNYNYLYRLFRKSNLKPFERDISTEIYQAGNKPCNGVAGFNCYFPLRSYSFRPTYG' \
                 'VGHQPYRVVVLSFELLHAPATVCGPKKST'
common_naming = 'notrim'
count_threshold = 3

main(RAW_SEQ_DIR, SAVE_DIR, lib_list, targets_list, labels_list, BA1_seq, common_naming, count_threshold)
