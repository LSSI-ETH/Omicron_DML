"""
Author: Beichen Gao, Jiami Han

Script to generate synthetic lineages based on GISAID observed mutational frequencies, and using ACE2 DML models
for prediction of next generation seeds.

Code is written to take input arguments compatible with Slurm Workload Manager on ETH Euler clusters. Can also be changed to run
locally and take local arguments
"""

import argparse
import math
import os
import random

import Levenshtein
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sc
import seaborn as sns
import warnings

from base_models import CNN_model_1D, MLP
from helpers import encode_onehot_padded


def create_parser():
    parser = argparse.ArgumentParser(description="Generator to generate synthetic RBD lineages",
                                     fromfile_prefix_chars='@')

    # ---data arguments
    parser.add_argument('--start_voc', type=str, default='BA.1',
                        help='name of initial variant used to generate lineages')
    parser.add_argument('--year', type=str, default='2022',
                        help='Year of GISAID sequences to use for probability matrix calculation')
    parser.add_argument('--threshold', type=str, default='hard',
                        help='type of threshold used to calculate probabilities, options: hard, soft')
    parser.add_argument('--subsample_size', type=int, default=500000,
                        help='number of sequences to generate per subsample')
    parser.add_argument('--set_number', type=int, default=1,
                        help='number of current set of sequences to be generated. This lets us automate generation of any number of sets of lineages')
    parser.add_argument('--random_seed', type=int, default=0,
                        help='random seed to be set, or if left as 0, random choice')
    parser.add_argument('--rounds', type=int, default=6,
                        help='number of rounds of mutations performed')
    parser.add_argument('--ace2_bind', type=str, default='true',
                        help='setting for whether ACE2 binding probabilities is used to determine seed sequences of next generation')
    parser.add_argument('--plot_mutation', type=str, default='false',
                        help='flag for whether if mutation heatmaps should be plotted for generated seqs')

    # DL Arguments
    parser.add_argument('--embedding', type=str, default='onehot',
                        help='embedding to use for sequences, options: onehot')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size during training')
    parser.add_argument("--base_model", type=str, default='cnn1d',
                        help='Type of model to use. Options: mlp, cnn1d')
    parser.add_argument('--rank_metric', type=str, default='mcc',
                        help='metric to use to get the top models, options: mcc, recall, precision')

    return parser


"""
if the n position is different it return n-1.
since the starting position is 331, instead of 1, we add 330 in the i.
"""


def find_differing_positions(str1, str2):
    return [i + 331 for i, (c1, c2) in enumerate(zip(str1, str2)) if c1 != c2]


def list_aa_mutations(ori_seq, loc, seq):
    original_loc = [x - 331 for x in loc]
    ori_aa = [ori_seq[i] for i in original_loc]
    mut_aa = [seq[i] for i in original_loc]
    spike_loc = [str(x + 331) for x in original_loc]
    aa_mutation = ["".join(items) for items in zip(ori_aa, spike_loc, mut_aa)]
    return aa_mutation


def calculate_mut(mut_round, seq, wt_seq, omi_seq):
    # calculate
    dis_omi = Levenshtein.distance(seq, omi_seq)
    dis_wt = Levenshtein.distance(seq, wt_seq)
    loc_omi = find_differing_positions(seq, omi_seq)
    loc_wt = find_differing_positions(seq, wt_seq)
    mut_omi = list_aa_mutations(omi_seq, loc_omi, seq)
    mut_wt = list_aa_mutations(wt_seq, loc_wt, seq)
    return mut_round, seq, dis_omi, dis_wt, loc_omi, loc_wt, mut_omi, mut_wt


def encode_data(df, encoding, batch_size, seq_column='aa'):
    """
        Encodes the data.
        Input:
            aa: list of aa sequences
    """
    x = []
    if encoding == "onehot":
        seqs = df[seq_column]
        for i in range(0, len(seqs), batch_size):
            if i % (batch_size * 10) == 0:
                print(f'Embedding sequences {i} / {len(seqs)}')
            batch_sequences = seqs[i: i + batch_size]
            outputs = encode_onehot_padded(batch_sequences)
            x.extend(outputs)
        x = np.stack(x)
    return x


def generator_omi(n_round, max_seq, model_list, model_type,
                  wt_seq, omi_seq, omi_seq_name,
                  embedding, batch_size=32, ace2_bind='true',
                  aa_probs=None, pos_probs=None):
    # define omicron_RBD: 331-531 (201aa):
    init_seq = omi_seq
    # define aa_list
    if aa_probs is not None:
        aa_list = list(aa_probs.columns)
    else:
        aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
                   'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    # define initiate seq_df
    init_round, init_seq, dis_omi, dis_wt, loc_omi, loc_wt, mut_omi, mut_wt = calculate_mut(0, init_seq, wt_seq,
                                                                                            omi_seq)
    init_df = pd.DataFrame({
        "round": init_round,
        "seq": init_seq,
        f"dis_{omi_seq_name}": dis_omi,
        "dis_wt": dis_wt,
        f"loc_{omi_seq_name}": [loc_omi],
        "loc_wt": [loc_wt],
        f"mut_{omi_seq_name}": [mut_omi],
        "mut_wt": [mut_wt],
        "pred": 1
    })
    final_df = init_df.copy()
    # max_seq control
    seq_per_round = max_seq / n_round
    # mutate generation
    for mut_round in range(1, n_round + 1):
        print(f'Generating for mutation round {mut_round}')
        round_df = init_df.copy()
        # define a new seq_df_last_round
        seq_df_last_round = final_df[final_df['round'] == mut_round - 1]
        n_parents = min(100, len(seq_df_last_round))

        for i in range(n_parents):  # loop 100 sequence from last round
            if i % 10 == 0:
                print(f'Generating for the {i}th parent of mutation round {mut_round}')
            if ace2_bind == 'true':
                parent_seq = seq_df_last_round.iloc[i]['seq']
            else:
                # choose random column from seq_df_last_round
                k = random.choice(range(len(seq_df_last_round)))
                parent_seq = seq_df_last_round.iloc[k]['seq']
            seq_per_parent = math.floor(seq_per_round / n_parents)

            # sample from 1~201 select location, 1 mutation each round. each round is 1 generation. each round only add new loc, no more mutation on the mutated place.
            excluded_index = [x - 331 for x in seq_df_last_round.iloc[i][f'loc_{omi_seq_name}']]
            loc = [x for x in range(201) if x not in excluded_index]
            if pos_probs is not None:
                # recalculate probability distribution for sampling
                # pos_probs_adj = np.delete(pos_probs, excluded_index) # has problems running on Euler some how. Replace with below line
                pos_probs_adj = [pos_probs[j] for j in loc]
                pos_probs_adj = [p / sum(pos_probs_adj) for p in pos_probs_adj]
                pos_probs_adj = np.array(pos_probs_adj)
            # start counts
            unique_seq_counter = 0
            failsafe_counter = 0
            max_failsafe = 10000
            while unique_seq_counter < seq_per_parent:
                # choose positions randomly
                if pos_probs is not None:
                    mut_loc = np.random.choice(a=loc, p=pos_probs_adj)
                else:
                    mut_loc = np.random.choice(a=loc)
                # remove aa from parent AA from selection pool
                aa_ori = parent_seq[mut_loc]
                aa_list_adj = [aa for aa in aa_list if aa != aa_ori]
                # get probabilities for position
                if aa_probs is not None:
                    aa_prob = aa_probs.iloc[mut_loc]
                    # drop probability at aa_ori
                    aa_prob_adj = aa_prob.drop(aa_ori)
                    aa_prob_adj = [p / sum(aa_prob_adj) for p in aa_prob_adj]
                    aa_mut = np.random.choice(a=aa_list_adj, p=aa_prob_adj)
                else:
                    aa_mut = np.random.choice(a=aa_list_adj)
                # replace loc-th aa in the seq
                mut_seq = parent_seq[:mut_loc] + aa_mut + parent_seq[(mut_loc + 1):]
                # check if generated sequence is unique
                if mut_seq not in round_df['seq'].values:
                    annotation = [score for score in calculate_mut(mut_round, mut_seq, wt_seq, omi_seq)]
                    round_df.loc[len(round_df)] = annotation + [np.nan]
                    unique_seq_counter += 1
                    failsafe_counter += 1
                    if unique_seq_counter % 100 == 0:
                        print(f'unique: {unique_seq_counter}, failsafe: {failsafe_counter}')
                else:
                    failsafe_counter += 1

                # check failsafe condition
                if failsafe_counter >= max_failsafe:
                    print("FAILSAFE ACTIVATED")
                    break

        # cleanup the round_df
        # drop initial sequence
        round_df = round_df[round_df['round'] != 0]
        round_df = round_df.drop_duplicates(subset='seq')
        # predict on the current round, add pred to a col.
        # print(f'Starting predictions for mutation round {mut_round}')
        encode_x = encode_data(round_df, encoding=embedding, batch_size=batch_size,
                               seq_column='seq')
        if model_type == 'mlp':
            # flatten encodings
            encode_x = encode_x.reshape(encode_x.shape[0], -1)

        # call the model and do prediction! -> go to another script to train and save model. script :hp, job name: save model, pick best from the metrics
        pred_cols = []
        for i in range(len(model_list)):
            clf_model = model_list[i]
            pred = clf_model.predict(encode_x)
            round_df[f'pred_{i}'] = pred
            pred_cols.append(f'pred_{i}')
        # calculate average predictions
        round_df['pred'] = round_df[pred_cols].mean(axis=1)
        # drop prediction columns
        round_df.drop(columns=pred_cols, inplace=True)
        # print(f'Mutation round {mut_round} completed. Moving on to next round')
        # add to full seq_df
        final_df = pd.concat([final_df, round_df])
        # rank seq by round and pred
        final_df = final_df.sort_values(by=['round', 'pred'], ascending=[False, False])
        final_df = final_df.drop_duplicates(subset='seq')
        # Reset the index of the sorted DataFrame
        final_df = final_df.reset_index(drop=True)
    return final_df


"""
make mut frequency matrices
"""
def calculate_aa_counts(df, seq_column):
    # Create a dictionary to store the frequencies for each position
    frequencies = {}

    # Iterate over each sequence in the specified column
    for sequence in df[seq_column]:
        # Iterate over each letter at each position in the sequence
        for position, aa in enumerate(sequence):
            # If the position is not already in the frequencies dictionary, add it
            if position not in frequencies:
                frequencies[position] = {}

            # If the letter is not already in the position's frequencies, add it
            if aa not in frequencies[position]:
                frequencies[position][aa] = 0

            # Increment the frequency count for the letter at the position
            frequencies[position][aa] += 1

    # Create a DataFrame from the frequencies dictionary
    frequencies_df = pd.DataFrame.from_dict(frequencies, orient='index')
    # Sort the columns alphabetically
    frequencies_df = frequencies_df.reindex(sorted(frequencies_df.columns), axis=1)
    # Fill any missing values with 0
    frequencies_df = frequencies_df.fillna(0)
    # # Normalize the frequencies as a proportion of the total count at each position
    # frequencies_df = frequencies_df.div(frequencies_df.sum(axis=1), axis=0)
    # Now sort by index to get by position
    frequencies_df = frequencies_df.sort_index()
    return frequencies_df


def remove_wt(df, wildtype_seq, replace_value=0):
    # set wildtype seq cells to 0
    for position, aa in enumerate(wildtype_seq):
        df.at[position, aa] = replace_value
    return df


def make_mut_probs(df, seq_column, wt_seq, stringency='hard'):
    """
    function to calculate aa and positional mutational probabilities
    from input GISAID data
    Parameters
    ----------
    df: df containing sequences from GISAID
    seq_column: (str) name of column where sequences are
    wt_seq: (str) wildtype sequence with aa's to remove from probabilities
    stringency: (str) type of stringency for processing aa_probabilities
        options: 'hard', 'soft' -
        'hard' will not allow any unseen mutations, and applies
        a log transform to reduce impact of high mutational counts to seen AA's to smooth
        out distribution before applying softmax.
        'soft' will simply set all seen mutations to a count of 1 before applying softmax
        which creates equal probabilities of all mutations seen in GISAID, while also allowing
        slightly smaller probabilities of mutating to any other unseen AA.

    Returns
    -------
    aa_mat_softmax: matrix of aa probabilities
    pos_probs_softmax: array of probabilities per position
    """

    # calculate count_matrix
    counts_mat = calculate_aa_counts(df, seq_column)
    # remove wt
    counts_mat = remove_wt(counts_mat, wt_seq)
    # drop X
    counts_mat = counts_mat.drop(columns=['X'])
    ##### AA Probability Matrix
    aa_mat = counts_mat.copy()
    # log transform counts to decrease effects of large counts
    aa_mat = np.log10(aa_mat)
    if stringency == 'soft':
        # set all columns with counts to 1
        aa_mat.replace([np.inf, -np.inf], 0, inplace=True)
    # tranform into probabilities
    aa_mat_softmax = sc.special.softmax(aa_mat, axis=1)
    aa_mat_softmax = pd.DataFrame(aa_mat_softmax, columns=aa_mat.columns)

    ##### Positional Probabilities
    # add all counts after removing WT
    pos_probs = counts_mat.sum(axis=1)
    # log transform counts, to decrease impact of large counts
    pos_probs_log = np.log10(pos_probs)
    pos_probs_softmax = sc.special.softmax(pos_probs_log)

    return aa_mat_softmax, pos_probs_softmax


"""
Function to load in models for generator
"""
def load_ace2_models(args, MODEL_DIR, SCORE_DIR):
    target = 'ACE2'

    WT_seq = "NITNLCPFDEVFNATRFASVYAWNRKRISNCVADYSVLYNLAPFFTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNKLDSKVSGNYNYLYRLFRKSNLKPFERDISTEIYQAGNKPCNGVAGFNCYFPLRSYSFRPTYGVGHQPYRVVVLSFELLHAPATVCGPKKST"
    WT_df = pd.DataFrame({'aa': WT_seq}, index=[0])
    init_x = encode_data(WT_df, encoding=args.embedding, batch_size=args.batch_size)
    if args.base_model == 'mlp':
        # flatten encodings
        init_x = init_x.reshape(init_x.shape[0], -1)

    # take metrics from top models selected from Top_models.py
    ##TODO: need to unify this file path with the metrics saved in Top_models.py
    top_models_df = pd.read_csv(f"{SCORE_DIR}/ACE2_{args.base_model}_test_{args.rank_metric}_top_model_metrics.csv",
                                index_col=0)
    top_models_df.reset_index(inplace=True)
    model_metrics = top_models_df.to_dict()

    # load in and initiate the top models
    model_list = []
    for i in range(len(top_models_df)):
        model_name = model_metrics['model_name'][i]
        if args.base_model == 'cnn1d':
            model = CNN_model_1D(stride=model_metrics['stride'][i],
                                 filter_num=model_metrics['filter_num'][i],
                                 padding=model_metrics['padding'][i],
                                 kernel_size=model_metrics['kernel_size'][i],
                                 dense_dim=model_metrics['dense_dim'][i],
                                 dense_dropout=model_metrics['dense_dropout'][i],
                                 pool_size=model_metrics['pool_size'][i],
                                 pool_stride=model_metrics['pool_stride'][i],
                                 residual_blocks=model_metrics['residual_blocks'][i],
                                 dilation_rate=model_metrics['dilation_rate'][i],
                                 regularizer_term=model_metrics['regularizer_term'][i])
        elif args.base_model == 'mlp':
            model = MLP(dense_dim=model_metrics['dense_dim'][i],
                        dense_dropout=model_metrics['dense_dropout'][i],
                        dense_layers=model_metrics['dense_layers'][i])

        model(init_x)
        model.load_weights(f'{MODEL_DIR}/{target}/weights/{model_name}')
        model_list.append(model)

    return model_list


"""
Define generator function
"""
def main_generator(args):
    """
    Define variables
    """
    WORK_DIR = os.getcwd()
    DATA_DIR = f"{WORK_DIR}/data"
    SCORE_DIR = f"{WORK_DIR}/voc_predictions"
    MODEL_DIR = f"{WORK_DIR}/models"
    SAVE_DIR = f"{WORK_DIR}/generated_seqs"
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    voc = args.start_voc
    year = args.year
    prob_threshold = args.threshold
    rounds = args.rounds
    seq_n = args.subsample_size
    set_n = args.set_number
    ace2_bind = args.ace2_bind
    if args.random_seed != 0:
        rand_seed = args.random_seed
    else:
        rand_seed = np.random.choice(range(1, 420))

    warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

    """
    Define main variant sequences to be used
    """
    wt_seq = "NITNLCPFDEVFNATRFASVYAWNRKRISNCVADYSVLYNLAPFFTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNKLDSKVSGNYNYLYRLFRKSNLKPFERDISTEIYQAGNKPCNGVAGFNCYFPLRSYSFRPTYGVGHQPYRVVVLSFELLHAPATVCGPKKST"
    voc_df = pd.read_csv(f'{DATA_DIR}/Canonical_spike_330530.csv', index_col=0)
    voc_seq = voc_df[voc_df['VariantName'] == voc]['sequence'].values[0]

    """
    Load in ACE2 models
    """
    ace2_models = load_ace2_models(args, MODEL_DIR, SCORE_DIR)

    """
    Calculate probability matrices and generate sets
    """
    if year != 'None':
        # load in gisaid data
        print(f'Loading in GISAID data from year {year} based on {voc}')
        gisaid = pd.read_csv(f"{DATA_DIR}/spikeprot{year}_all_330530.csv",
                             encoding='latin-1')
        gisaid['length'] = gisaid['sequence'].str.len()

        # filter for correct length
        gisaid = gisaid[gisaid['length'] == 201]
        print(f'Now calculating {prob_threshold} threshold probabilities from GISAID sequences')
        aa_mat, pos_mat = make_mut_probs(gisaid, 'sequence', voc_seq, stringency=prob_threshold)
        # save matrices
        aa_mat.to_csv(f"{SAVE_DIR}/{year}_{prob_threshold}threshold_aa_matrix.csv")
        pos_mat_df = pd.DataFrame(pos_mat, columns=['probability'])
        pos_mat_df.to_csv(f"{SAVE_DIR}/{year}_{prob_threshold}threshold_pos_matrix.csv")

        FINAL_DIR = f"{SAVE_DIR}/ace2_{ace2_bind}/{year}/{seq_n}"
        if not os.path.exists(FINAL_DIR):
            os.makedirs(FINAL_DIR)

        # start to generate set given seed
        # fix random seed
        np.random.seed(rand_seed)
        print(f'Generating sequences for set {set_n}')
        n_round = rounds
        seq_df = generator_omi(n_round, seq_n, ace2_models, args.base_model,
                               wt_seq, voc_seq, voc,
                               embedding=args.embedding, batch_size=args.batch_size, ace2_bind=ace2_bind,
                               aa_probs=aa_mat, pos_probs=pos_mat)

    else:
        print(f'Generating random sequences based on {voc}')
        FINAL_DIR = f"{SAVE_DIR}/ace2_{ace2_bind}/{year}/{seq_n}"
        if not os.path.exists(FINAL_DIR):
            os.makedirs(FINAL_DIR)
        # fix random seed
        np.random.seed(rand_seed)
        print(f'Generating sequences for set {set_n}')
        n_round = rounds
        seq_df = generator_omi(n_round, seq_n, ace2_models, args.base_model,
                               wt_seq, voc_seq, voc,
                               embedding=args.embedding, batch_size=args.batch_size, ace2_bind=ace2_bind)

    seq_df.to_csv(
        f'{FINAL_DIR}/{voc}_generated_{prob_threshold}threshold_{n_round}rounds_set{set_n + 1}_seed{rand_seed}.csv')
    """
    If desired, plot heatmap of mutation distribution per position
    """
    if args.plot_mutation == 'true':
        # check distribution of mutations per position
        seq_aa_counts = calculate_aa_counts(seq_df, 'seq')
        seq_aa_counts = remove_wt(seq_aa_counts, voc_seq)
        seq_aa_log = np.log(seq_aa_counts)
        seq_aa_log.replace([np.inf, -np.inf], np.nan, inplace=True)

        seq_aa_log_t = seq_aa_log.transpose()
        alt_aa_order = ['R', 'K', 'H', 'D', 'E', 'Q', 'N', 'S', 'T',
                        'Y', 'W', 'F', 'A', 'I', 'L', 'M', 'V', 'G',
                        'P', 'C']
        seq_aa_log_t = seq_aa_log_t.reindex(alt_aa_order)
        # sns.set(rc={'figure.figsize': (50, 5)})
        # fig, ax = plt.subplots(figsize=(50, 5))
        sns.set_context("paper", rc={"font.size": 15, "axes.titlesize": 15, "axes.labelsize": 10})
        plt.figure(figsize=(15, 4))
        palette = sns.diverging_palette(220, 20, as_cmap=True)
        # ax = sns.heatmap(rel_freqs_log_t, cmap=palette, vmin=-3.5, vmax=3.5,
        #                  cbar_kws={'orientation':'vertical', 'pad':0.02}, center=0)
        ax = sns.heatmap(seq_aa_log_t, cmap=palette, center=0,
                         cbar_kws={'orientation': 'vertical', 'pad': 0.02,
                                   'label': 'relative freq (log)'},
                         xticklabels=10)

        ax.set_facecolor('grey')
        # set labels
        x_labels = range((331), (532), 10)
        ax.set(xlabel='Position', ylabel='Residue')
        ax.set_xticklabels(x_labels, size=10)
        ax.set_yticklabels(list(seq_aa_log_t.index), size=10)
        # colourbar labels
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=8)
        ax.figure.axes[-1].yaxis.label.set_size(10)
        plt.title(f'Generated Seqs AA mutation relative frequencies')
        plt.tight_layout()
        plt.savefig(
            f"{FINAL_DIR}/{voc}_generated_{prob_threshold}threshold_{n_round}rounds_set{set_n + 1}_seed{rand_seed}_probability_heatmap.png",
            dpi=300, format='png', bbox_inches='tight')
        plt.show()
        plt.close()


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main_generator(args)
