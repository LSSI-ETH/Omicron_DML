"""
Author: Beichen Gao

RBD mAb breadth predictions on synthetic lineages generated through lineage_generator.py (but also compatible with other published
datasets if RBD sequences have been extracted)

So this script will:
- Navigate to folder containing 10x sets of sample data generated with a specific sample size each
- Run predictions using all top mAb models
- Filter for ACE2 binding sequences only, then evaluate proportion of sequences covered by individual and combinations of mAbs
- Output scores into folder for figures

Code is written to take input arguments compatible with Slurm Workload Manager on ETH Euler clusters. Can also be changed to run
locally and take local arguments
"""

"""
Import all required packages
"""
import argparse
import math
import os
import random
import time

import numpy as np
import pandas as pd

from utils.dl_model_utils import CNN_model_1D, MLP
from utils.helpers import encode_onehot_padded

timestr = time.strftime("%Y%m%d-%H")

rand_state = 8

def create_parser():
    parser = argparse.ArgumentParser(description="Breadth predictions on generated sequences from Omicron library", fromfile_prefix_chars='@')

    # ---data arguments
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='location of labelled datasets for model training')
    parser.add_argument('--target', type=str, default='ACE2',
                        help='name of target to run predictions for')
    parser.add_argument('--start_voc', type=str, default='BA.2',
                        help='initial VOC used to generate lineages')
    parser.add_argument('--year', type=str, default='2022',
                        help='Year of GISAID sequences to use for prediction')
    parser.add_argument('--threshold', type=str, default='hard',
                        help='prediction matrix used')
    parser.add_argument('--subsample_size', type=int, default=10000,
                        help='number of sequences in subsample used')
    parser.add_argument('--subsample_number', type=int, default=1,
                        help='specific generated subsample set to use')
    parser.add_argument('--test_seed', type=int, default=1,
                        help='seed of models to take from')
    parser.add_argument("--minority_ratio", type=float, default=0.25,
                        help='minority ratio selected for best outcomes from HP optimization')
    parser.add_argument("--n_top_models", type=int, default=3,
                        help='number of top models to use for predictions')
    parser.add_argument("--model_depth", type=int, default=2,
                        help='depth of model used')
    parser.add_argument('--rank_metric', type=str, default='mcc',
                        help='metric to use to get the top models, options: mcc, recall, precision')
    parser.add_argument('--rounds', type=int, default=6,
                        help='number of rounds of mutations performed')
    parser.add_argument('--bind_threshold', type=float, default=0.75,
                        help='threshold for positive labels')
    parser.add_argument('--esc_threshold', type=float, default=0.25,
                        help='threshold for escape labels')
    parser.add_argument('--ace2_bind', type=str, default='true',
                        help='whether if the dataset used has been tested for ace2_binding. Options: true, false')
    # DL Arguments (mostly used to get the right score dataframes)
    parser.add_argument('--embedding', type=str, default='onehot',
                        help='embedding to use for sequences, options: onehot')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size during training')
    parser.add_argument("--base_model", type=str, default='cnn',
                        help='Type of model to use. Options: mlp, cnn')
    parser.add_argument("--prediction_method", type=str, default='majority_voting',
                        help='Method used to predict final labels. Options: majority_voting, average')
    return parser



"""
Main Functions
"""

"""
Function that saves each subsample split, and the number of binders and escapes
"""

def subsample_info(df, target_list, subsample_name, subsample_size, SAVE_DIR):
    """
    Function to save and evaluate the binding/nonbinding distribution of all mAbs of interest within the
    subsample of the overall dataframe

    Parameters
    ----------
    df : DataFrame
        Dataframe, with columns indicating predictions of each mAb.
    mAbs : List
        List of names of the mAbs of interest, corresponding to columns in the df.
    subsample_name : String
        Name of the subsample, usually "R1/R2/etc".

    Returns
    -------
    None.

    """
    # Now make a df of binders and non-binders of each mAb
    binders_list = []
    nonbinders_list = []
    unsure_list = []

    for target in target_list:
        binders_list.append(len(df[df[target] == 1]) / len(df))
        nonbinders_list.append(len(df[df[target] == 0]) / len(df))
        unsure_list.append(len(df[df[target] == 0.5])/len(df))
    info_df = pd.DataFrame()
    info_df['target'] = target_list
    info_df['Binders'] = binders_list
    info_df['Non-Binders'] = nonbinders_list
    info_df['Unsure'] = unsure_list
    # Now save info_df
    info_df.to_csv(f"{SAVE_DIR}/{subsample_size}_seq_subsample_{subsample_name}_info.csv")


"""
Function that calculates overlap, and additive percentage of sequences for combinations of mAbs
"""

def prediction_evaluation(df, mAb1_list, mAb2_list):
    # iterate through combinations of the mAbs and calculate sequence overlap percentage
    abs_overlap_perc_list = []
    rel_overlap_perc_list = []
    additive_perc_list = []
    for i in range(len(mAb1_list)):
        mAb1 = mAb1_list[i]
        mAb2 = mAb2_list[i]
        # First, filter for ACE2 binding
        ace2_df = df[df['ACE2'] == 1]
        # Calculate overlap
        overlap_seqs = ace2_df[(ace2_df[mAb1] == 1) & (ace2_df[mAb2] == 1)]
        mAb1_total_seqs = ace2_df[ace2_df[mAb1] == 1]
        if len(overlap_seqs) == 0:
            abs_overlap = 0
            rel_overlap = 0
        else:
            # overlap_perc1 = len(overlap_seqs)/len(df[df[mAb1] == 1])
            # overlap_perc2 = len(overlap_seqs)/len(df[df[mAb2] == 1])
            # best_overlap = max([overlap_perc1, overlap_perc2])
            abs_overlap = len(overlap_seqs) / len(ace2_df)
            rel_overlap = len(overlap_seqs) / len(mAb1_total_seqs)

        # overlap_perc_list.append(best_overlap)
        # overlap_perc_list.append(overlap_perc1)
        abs_overlap_perc_list.append(abs_overlap)
        rel_overlap_perc_list.append(rel_overlap)

        # Calculate additive
        additive_seqs = ace2_df[(ace2_df[mAb1] == 1) | (ace2_df[mAb2] == 1)]
        if len(additive_seqs) == 0:
            additive_perc = 0
        else:
            additive_perc = len(additive_seqs) / len(ace2_df)
        additive_perc_list.append(additive_perc)

    return abs_overlap_perc_list, rel_overlap_perc_list, additive_perc_list


"""
Functions for importing the top models, initating them, and loading in the weights
"""

def encode_data(df, embedding, batch_size, seq_column='aa'):
    """
        Encodes the data.
        Input:
            aa: list of aa sequences
    """
    x = []
    if embedding == "onehot":
        seqs = df[seq_column]
        for i in range(0, len(seqs), batch_size):
            if i % (batch_size * 10) == 0:
                print(f'Embedding sequences {i} / {len(seqs)}')
            batch_sequences = seqs[i: i + batch_size]
            outputs = encode_onehot_padded(batch_sequences)
            x.extend(outputs)
        x = np.stack(x)
    elif embedding == "esm":
        # find name of the column containing esm embeddings
        esm_col = [col for col in df.columns if 'esm' in col]
        esm_col_name = esm_col[0]
        embeddings = df[esm_col_name].values
        final_embeddings = []
        for i in range(0, len(embeddings), batch_size):
            if i % (batch_size * 10) == 0:
                print(f'Embedding sequences {i} / {len(embeddings)}')
            batch = embeddings[i: i + batch_size]
            for seq in batch:
                embed = seq[1:-1]
                embed = embed.split(', ')
                embed = [float(x) for x in embed]
                embed = np.array(embed)
                final_embeddings.append(embed)
        x = np.stack(final_embeddings)
    return x


def initiate_models(target, model_type, rank_metric, test_seed, model_depth, METRICS_DIR, MODEL_DIR, embedding, batch_size=32):
    WT_seq = "NITNLCPFDEVFNATRFASVYAWNRKRISNCVADYSVLYNLAPFFTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNKLDSKVSGNYNYLYRLFRKSNLKPFERDISTEIYQAGNKPCNGVAGFNCYFPLRSYSFRPTYGVGHQPYRVVVLSFELLHAPATVCGPKKST"
    WT_df = pd.DataFrame({'aa': WT_seq}, index=[0])
    init_x = encode_data(WT_df, embedding=embedding, batch_size=batch_size)
    if model_type == 'mlp':
        # flatten encodings
        init_x = init_x.reshape(init_x.shape[0], -1)

    # load in models from top metrics df
    top_models_df = pd.read_csv(
        f"{METRICS_DIR}/{target}_{model_type}_seed{test_seed}_depth{model_depth}_test_{rank_metric}_top_model_metrics.csv",
        index_col=0)
    # split into lib1 and lib2
    libraries = ['Lib1', 'Lib2']
    model_list = []
    for lib in libraries:
        models_df = top_models_df[top_models_df['library'] == lib]
        models_df.reset_index(inplace=True)
        model_metrics = models_df.to_dict()
        for i in range(len(models_df)):
            model_name = model_metrics['model_name'][i]
            if model_type == 'cnn':
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
            elif model_type == 'mlp':
                model = MLP(dense_dim=model_metrics['dense_dim'][i],
                            dense_dropout=model_metrics['dense_dropout'][i],
                            dense_layers=model_metrics['dense_layers'][i])

            # initiate model by putting through init_x
            model(init_x)
            model.load_weights(f"{MODEL_DIR}/{target}/weights/{model_name}").expect_partial()
            model_list.append(model)

    return model_list

def ensemble_predict(target, model_list, subsample_name,
                     df, embed_x, seq_column, batch_size, FULL_PREDS_SAVE_DIR,
                     bind_threshold=0.5, esc_threshold=None):
    pred_df = df.copy()
    pred_df = pred_df[[seq_column]]
    pred_cols = []
    for i in range(len(model_list)):
        clf_model = model_list[i]
        num_samples = len(df)
        num_batches = math.ceil(num_samples / batch_size)

        preds = []
        for batch_idx in range(num_batches):
            # get batch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_x = embed_x[start_idx:end_idx]
            # predict on batch
            batch_pred = clf_model.predict(batch_x)
            # batch_pred = pd.Series(batch_pred)
            if batch_idx == 0:
                preds = batch_pred
            else:
                preds = np.concatenate((preds, batch_pred))
        # preds = clf_model.predict(embed_x)
        pred_df[f'{target}_pred{i}'] = preds
        pred_cols.append(f'{target}_pred{i}')
    # calculate average predictions
    final_pred = pred_df[pred_cols].mean(axis=1)
    pred_df[f'{target}'] = final_pred
    # save prediction df
    if not os.path.exists(f"{FULL_PREDS_SAVE_DIR}/{subsample_name}"):
        os.makedirs(f"{FULL_PREDS_SAVE_DIR}/{subsample_name}")

    pred_df.to_pickle(f"{FULL_PREDS_SAVE_DIR}/{subsample_name}/{target}_threshold{bind_threshold}_predictions.pkl")
    # add final prediction to main df
    df[f'{target}'] = final_pred
    if bind_threshold is not None:
        final_pred[final_pred >= bind_threshold] = 1
        if esc_threshold is not None:
            final_pred[final_pred <= esc_threshold] = 0
            final_pred[(final_pred > esc_threshold) & (final_pred < bind_threshold)] = 0.5
        else:
            final_pred[final_pred < bind_threshold] = 0
        df[f'{target}'] = final_pred
    return df


def ensemble_vote(target, model_list, subsample_name,
                  df, embed_x, seq_column, batch_size, FULL_PREDS_SAVE_DIR,
                  bind_threshold=0.5, esc_threshold=0.5, to_int=True):
    pred_df = df.copy()
    pred_df = pred_df[[seq_column]]
    # split models into library 1 and library 2
    lib1_models = model_list[0:3]
    lib2_models = model_list[3:]
    # lib1 first
    lib1_cols = []
    lib2_cols = []
    for i in range(len(lib1_models)):
        clf_model = lib1_models[i]
        num_samples = len(df)
        num_batches = math.ceil(num_samples / batch_size)
        preds = []
        for batch_idx in range(num_batches):
            # get batch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_x = embed_x[start_idx:end_idx]
            # predict on batch
            batch_pred = clf_model.predict(batch_x)
            # batch_pred = pd.Series(batch_pred)
            if batch_idx == 0:
                preds = batch_pred
            else:
                preds = np.concatenate((preds, batch_pred))
        # get model labels
        preds_str = preds.copy()
        preds_str[preds > bind_threshold] = 1
        preds_str[preds < esc_threshold] = 0
        preds_str[(preds < bind_threshold) & (preds > esc_threshold)] = 0.5
        pred_df[f'{target}_Lib1_pred{i}'] = preds_str
        lib1_cols.append(f'{target}_Lib1_pred{i}')
        # ======================= Majority Vote per library =======================
        pred_df[f'{target}_Lib1'] = pred_df[lib1_cols].mode(axis=1)[0]

    for j in range(len(lib2_models)):
        clf_model = lib2_models[j]
        num_samples = len(df)
        num_batches = math.ceil(num_samples / batch_size)
        preds = []
        for batch_idx in range(num_batches):
            # get batch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_x = embed_x[start_idx:end_idx]
            # predict on batch
            batch_pred = clf_model.predict(batch_x)
            # batch_pred = pd.Series(batch_pred)
            if batch_idx == 0:
                preds = batch_pred
            else:
                preds = np.concatenate((preds, batch_pred))
        # get model labels
        preds_str = preds.copy()
        preds_str[preds > bind_threshold] = 1
        preds_str[preds < esc_threshold] = 0
        preds_str[(preds < bind_threshold) & (preds > esc_threshold)] = 0.5
        pred_df[f'{target}_Lib2_pred{j}'] = preds_str
        lib2_cols.append(f'{target}_Lib2_pred{j}')
        # ======================= Majority Vote per library =======================
        pred_df[f'{target}_Lib2'] = pred_df[lib2_cols].mode(axis=1)[0]
    # ======================= Final majority vote =======================
    lib_col_names = [f'{target}_Lib1', f'{target}_Lib2']

    def set_final_prediction(row):
        """
        Taken from ChatGPT
        sets final prediction to escape if either libraries predict escape, unsure if
        no escape prediction, but binding is unsure
        Parameters
        ----------
        row

        Returns
        -------

        """
        libraries_predictions = row[lib_col_names].tolist()
        bind_count = libraries_predictions.count(1)
        esc_count = libraries_predictions.count(0)
        unsure_count = libraries_predictions.count(0.5)

        if esc_count == 0 and unsure_count == 0:
            return 1
        elif esc_count == 0 and unsure_count > 0:
            return 0.5
        else:
            return 0  # or any other default value for ties

    pred_df['final_score'] = pred_df.apply(set_final_prediction, axis=1)
    if to_int == False:
        int_pred = pred_df['final_score']
        int_pred[int_pred == 1] = 'bind'
        int_pred[int_pred == 0.5] = 'unsure'
        int_pred[int_pred == 0] = 'esc'
        pred_df['final_score'] = int_pred
    pred_df.reset_index(drop=True, inplace=True)
    if not os.path.exists(f"{FULL_PREDS_SAVE_DIR}/{subsample_name}"):
        os.makedirs(f"{FULL_PREDS_SAVE_DIR}/{subsample_name}")
    pred_df.to_pickle(f"{FULL_PREDS_SAVE_DIR}/{subsample_name}/{target}_threshold{bind_threshold}_predictions.pkl")

    df[f'{target}'] = pred_df['final_score']
    return df


"""
Wrap all functions together into a final prediction and analysis workflow
"""

def main(args):

    # Import args
    year = args.year
    start_voc = args.start_voc
    subsample_size = args.subsample_size
    sample_number = args.subsample_number
    rounds = args.rounds
    model_type = args.base_model
    rank_metric = args.rank_metric
    embedding = args.embedding
    batch_size = args.batch_size
    type_threshold = args.threshold
    test_seed = args.test_seed
    model_depth = args.model_depth
    bind_threshold = args.bind_threshold
    esc_threshold = args.esc_threshold
    prediction_method = args.prediction_method
    ace2_bind = args.ace2_bind

    # ======================= Set directories =======================
    WORK_DIR = os.getcwd()
    METRICS_DIR = f"{WORK_DIR}/voc_predictions/ratio{args.minority_ratio}/{args.n_top_models}/{args.rank_metric}/"
    MODEL_DIR = f"{WORK_DIR}/models"
    DATA_DIR = f"{WORK_DIR}/generated_seqs/ace2_{args.ace2_bind}/{args.year}/{args.subsample_size}"
    RUN_NAME = f"ace2_{args.ace2_bind}/{args.rank_metric}/{args.year}/{args.start_voc}/{args.subsample_size}/{args.threshold}/{args.prediction_method}"
    SAVE_DIR = f"{WORK_DIR}/generated_predictions/{RUN_NAME}"
    FULL_PREDS_SAVE_DIR = f"{WORK_DIR}/full_preds/{RUN_NAME}"
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # fix seed
    np.random.seed(args.subsample_number)
    random.seed(args.subsample_number)

    # ======================= Create all score dfs to be populated later =======================
    if args.target != 'all':
        target_list = [args.target]
    else:
        target_list = ['ACE2', 'A23581', 'Cov22196', 'ZCB11', '2_7', 'BRII198', 'ADG20', 'S2X259', 'S2H97']

    target1_list = []
    for target in target_list:
        names_list = [target] * len(target_list)
        target1_list.extend(names_list)
    target2_list = target_list * len(target_list)

    abs_overlap_scores_df = pd.DataFrame()
    abs_overlap_scores_df['mAb1'] = target1_list
    abs_overlap_scores_df['mAb2'] = target2_list

    rel_overlap_scores_df = pd.DataFrame()
    rel_overlap_scores_df['mAb1'] = target1_list
    rel_overlap_scores_df['mAb2'] = target2_list

    additive_scores_df = pd.DataFrame()
    additive_scores_df['mAb1'] = target1_list
    additive_scores_df['mAb2'] = target2_list

    # ======================= import in generated subsamples with wildcard for random-seed =======================
    sample_name = f"{start_voc}_generated_{type_threshold}threshold_{rounds}rounds_set{sample_number}_"
    print(f"Importing data for {sample_name}")
    files = os.listdir(DATA_DIR)
    match_files = []
    for x in files:
        if x.endswith('.csv'):
            if sample_name in x:
                match_files.append(x)
    # read in matching file
    print(f"Reading data for {match_files[0]}")
    subsample = pd.read_csv(f"{DATA_DIR}/{match_files[0]}")
    # drop duplicates
    subsample=subsample.drop_duplicates(subset='seq')
    subsample=subsample.reset_index(drop=True)

    # ======================= Run predictions using all models =======================
    # embed sequences first
    embed_x = encode_data(subsample, embedding=embedding, batch_size=batch_size,
                          seq_column='seq')
    if embedding == 'onehot':
        if model_type == 'mlp':
            embed_x = embed_x.reshape(embed_x.shape[0], -1)

    for target in target_list:
        # load in models
        model_list = initiate_models(target, model_type=model_type, rank_metric=rank_metric,
                                     test_seed=test_seed,
                                     METRICS_DIR=METRICS_DIR, MODEL_DIR=MODEL_DIR, embedding=embedding)
        if prediction_method == 'majority_voting':
            subsample = ensemble_vote(target, model_list, sample_number, subsample, embed_x, seq_column='seq',
                                      batch_size=batch_size, FULL_PREDS_SAVE_DIR=FULL_PREDS_SAVE_DIR,
                                      bind_threshold=bind_threshold, esc_threshold=esc_threshold)
        else:
            subsample = ensemble_predict(target, model_list, sample_number, subsample, embed_x, seq_column='seq',
                                         batch_size=batch_size, FULL_PREDS_SAVE_DIR=FULL_PREDS_SAVE_DIR,
                                         bind_threshold=bind_threshold)
    subsample_info(subsample, target_list, sample_number, subsample_size, SAVE_DIR=SAVE_DIR)
    # calculate overlap/additive scores
    abs_overlap_scores, rel_overlap_scores, additive_scores = prediction_evaluation(subsample, target1_list,
                                                                                    target2_list)
    abs_overlap_scores_df[sample_number] = abs_overlap_scores
    rel_overlap_scores_df[sample_number] = rel_overlap_scores
    additive_scores_df[sample_number] = additive_scores

    abs_overlap_scores_df.to_csv(
        f"{SAVE_DIR}/{subsample_size}_sample{sample_number}_abs_overlap_scores_{type_threshold}{bind_threshold}_df.csv")
    rel_overlap_scores_df.to_csv(
        f"{SAVE_DIR}/{subsample_size}_sample{sample_number}_rel_overlap_scores_{type_threshold}{bind_threshold}_df.csv")
    additive_scores_df.to_csv(
        f"{SAVE_DIR}/{subsample_size}_sample{sample_number}_total_additive_scores_{type_threshold}{bind_threshold}_df.csv")


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)

