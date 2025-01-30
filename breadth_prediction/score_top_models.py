"""
Author: Beichen Gao

Analysing output from DL HP Search runs

Includes:
- Sorting models by best score
- Plotting model performance
- Retrieving predictions on validation VoC subset

"""

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

"""
Importing all stored metrics from previous runs
"""
WORK_DIR = os.getcwd() # set WORK_DIR to current directory by default
targets = ['ACE2']
threshold = 2
info_df = pd.DataFrame()

for target in targets:
    metrics_dir = (f'{WORK_DIR}/metrics/{target}')
    if os.path.exists(metrics_dir):
        for x in os.listdir(f'{metrics_dir}/'):
            if x.endswith(".csv"):
                model_name = x.removesuffix('_metrics.csv')
                df = pd.read_csv(f'{metrics_dir}/{x}', index_col=0)
                info_df[model_name] = df.iloc[:, 0]  # move first column with metrics into info_df

info_df = info_df.transpose().reset_index()
info_df.rename(columns={'index': 'model_name'}, inplace=True)
# turn all cols except dataset into numeric
num_cols = info_df.columns.drop(
    ['target', 'model_name', 'run_name', 'library', 'rejection_sampling', 'padding', 'base_model'])
info_df[num_cols] = info_df[num_cols].apply(pd.to_numeric, errors='coerce')

# Add a column for "model_depth" that adds dense_layers or residual_blocks depending on model
cnn_df = info_df[info_df['base_model'] == 'cnn1d']
cnn_df['model_depth'] = cnn_df['residual_blocks']
mlp_df = info_df[info_df['base_model'] == 'mlp']
mlp_df['model_depth'] = mlp_df['dense_layers']

# recreate info_df
info_df = pd.concat([cnn_df, mlp_df])

# isolate zeros
info_df_zeroes = info_df[info_df['test_mcc'] == 0]

# drop all rows with test_mcc == 0
info_df = info_df[info_df['test_mcc'] != 0]

"""
Plot scores with seaborn and visualise
"""
# Create directory to save figures
FIG_DIR = f"{WORK_DIR}/figures"
if not os.path.exists(FIG_DIR):
    os.makedirs(FIG_DIR)

# Set the main parameters to filter by/plot
plot_score = 'test_mcc'
col_var = 'library'
row_var = 'dense_dim'
hue_var = 'model_depth'
filter_var = 'test_seed'
filter_value = 1
base_model = 'cnn'

# Plot facetgrid of model performance given parameters
if filter_var is not None:
    info_df_filtered = info_df[info_df[filter_var] == filter_value]
    a = sns.catplot(info_df_filtered, x='target', y=plot_score, col=col_var,
                    row=row_var, hue=hue_var, kind='box')
else:
    a = sns.catplot(info_df, x='target', y=plot_score, col=col_var,
                    row=row_var, hue=hue_var, kind='box')
a.set(ylim=(0.5, 1.0))
a.set_xticklabels(rotation=90)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/{plot_score}_by_{col_var}_{row_var}_{hue_var}_{filter_var}{filter_value}.png', dpi=300, format='png')
plt.show()
plt.close()

"""
If desired, plot all metrics together into boxplot
"""
id_list = ['library', 'target', 'test_seed', 'model_depth', 'epoch']
info_df_melt = pd.melt(info_df, id_vars=id_list, value_vars=['test_mcc', 'test_precision', 'test_recall', 'test_acc'])
info_df_melt.rename(columns={'variable':'metric', 'value':'score'}, inplace=True)

a = sns.catplot(info_df_melt, x='target', y='score', col=col_var, row='epoch',
                hue='metric', kind='box')

a.set(ylim=(0.5, 1.0))
a.set_xticklabels(rotation=90)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/{base_model}_{plot_score}_all.png', dpi=300, format='png')
plt.show()
plt.close()

"""
Functions to take predictions of top models
"""

def top_model_voc_predictions(DATA_DIR, PRED_DIR, SAVE_DIR, metrics_df, target, libraries,
                              base_model, test_seed, model_depth, rank_metric, n_models=3):
    """

    Parameters
    ----------
    DATA_DIR: (str) Path to directory containing data used for model training
    PRED_DIR: (str) Path to directory containing voc predictions
    metrics_df: (df) DataFrame containing all model metrics
    target: (str) target on which predictions have been made
    libraries: (list) of strings indicating libraries (ex. Lib1, Lib2)
    rank_metric: (str) name of column containing the metric to rank all models by
    SAVE_DIR: (str) Path to directory to save top model predictions
    n_models: (int) number of top models to take predictions from

    Returns
    -------
    top_metrics_df: (df) DataFrame containing all metrics of top models selected
    top_predictions_df: (df) DataFrame containing all top model predictions
    """
    # list out all the columns containing crucial information for voc predictions
    voc_info_cols = ['ID', 'VariantClass', 'VariantName']
    # find top n models
    target_df = metrics_df[metrics_df['target'] == target]
    target_df = target_df[target_df['base_model'] == base_model]
    target_df = target_df[target_df['test_seed'] == test_seed]
    if model_depth is not None:
        target_df = target_df[target_df['model_depth'] == model_depth]
    # create lists to store model names, metrics, and predictions
    top_models_list = []
    top_metrics_list = []
    # read in original voc df and append scores onto it
    top_predictions_df = pd.read_csv(f"{DATA_DIR}/VOC_Spike_all.csv",
                                     usecols=voc_info_cols)
    for lib in libraries:
        lib_df = target_df[target_df['library'] == lib]
        # sort by ranking score
        lib_df = lib_df.sort_values(rank_metric, ascending=False)
        # get names of top n models
        top_models = list(lib_df['model_name'][:n_models])
        top_models_list = top_models_list + top_models
        # store metrics for top n models
        top_metrics_list.append(lib_df[lib_df['model_name'].isin(top_models)])
        for i in range(len(top_models)):
            tmodel = top_models[i]
            keep_cols = voc_info_cols + [tmodel]
            metrics_filename = f"{PRED_DIR}/{target}/{tmodel}_test_variants.csv"
            if os.path.exists(metrics_filename):
                voc_df = pd.read_csv(f"{PRED_DIR}/{target}/{tmodel}_test_variants.csv",
                                     usecols=keep_cols)
                top_predictions_df[tmodel] = voc_df[tmodel]
            else:
                top_predictions_df[tmodel] = np.nan
    # Calculate mean across all top models and save
    top_predictions_df['avg_score'] = top_predictions_df[top_models_list].mean(axis=1)
    top_predictions_df.reset_index(drop=True, inplace=True)
    top_predictions_df.to_csv(f"{SAVE_DIR}/{target}_{base_model}_seed{test_seed}_depth{model_depth}_{rank_metric}_top_model_voc_predictions.csv")
    # save final metrics as well
    top_metrics_df = pd.concat(top_metrics_list)
    top_metrics_df.to_csv(f"{SAVE_DIR}/{target}_{base_model}_seed{test_seed}_depth{model_depth}_{rank_metric}_top_model_metrics.csv")
    return top_predictions_df, top_metrics_df


def plot_voc_predictions(SAVE_DIR, pred_df, target, class_id='VariantClade', focus_voc=None, canon_df=None):
    """

    Parameters
    ----------
    SAVE_DIR: (str) Path to directory to save figures
    pred_df: (df) DataFrame containing predictions of top models
    target: (str) target on which predictions are made
    class_id: (str) which definition of VoC is used to make boxplots
    rank_metric: (str) name of metric used to select top models
    focus_voc: (str) name of particular VoC lineage class to focus on plotting

    Returns
    -------
    None
    """
    # filter using a set of canonical sequences dataframe, if required
    if canon_df is not None:
        # drop VariantClass and VariantName columns
        pred_df.drop(['VariantClass', 'VariantName'], axis=1, inplace=True)
        # then merge with canon_df
        pred_df = pd.merge(pred_df, canon_df, on=['ID', 'sequence'], how='inner')
    # Sort by focus voc
    pred_df = pred_df[pred_df['VariantClass'] != 'Pango']
    if focus_voc is not None:
        pred_df = pred_df[pred_df['VariantClass'] == focus_voc]
    # Create "VariantClade" column based on "VariantName"
    pred_df['VariantClade'] = pred_df['VariantName'].str.split(".").str[:2].str.join(".")
    # Create boxplot
    a = sns.boxplot(pred_df, x=class_id, y='avg_score')
    a.set_title(f"{target} Top Model Predictions on VoC grouped by {class_id}")
    a.set_xticklabels(a.get_xticklabels(), rotation=90)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/{target}_{focus_voc}voc_{class_id}_predictions.png",
                dpi=300, format='png')
    plt.show()
    plt.close()

"""
Take top N models ranked by specific metric, plot out predictions on VOC, and store all 
predictions and scores into a separate dataframe to be used for future analysis steps (ex. Breadth)
"""
# set variables (these can always be changed depending on your own workflow)
DATA_DIR = f"{WORK_DIR}/data" # these may have changed depending on the variables set during DL_models run
PRED_DIR = f"{WORK_DIR}/voc"
targets = ['ACE2']
libs_list = ['Lib1', 'Lib2']
base_models = ['cnn']
model_depths = [2, None]
lr = [0.0001]
test_seeds = [1]
rank_metrics = ['test_mcc', 'test_recall', 'test_precision']
n_top_models = [3]
voc_plot_class = 'VariantName'
main_voc = 'Omicron'
minority_ratio = 0.25

for seed in test_seeds:
    info_df_subset = info_df[info_df['minority_ratio'] == minority_ratio]
    info_df_subset = info_df_subset[info_df_subset['test_seed'] == seed]
    for base_model in base_models:
        for depth in model_depths:
            for rank_metric in rank_metrics:
                for n_models in n_top_models:
                    # create save path
                    PRED_SAVE_DIR = f"{WORK_DIR}/voc_predictions/ratio{minority_ratio}/{n_models}/{rank_metric}"
                    if not os.path.exists(PRED_SAVE_DIR):
                        os.makedirs(PRED_SAVE_DIR)
                    # create final dataframe for all scores
                    total_voc_df = pd.DataFrame()
                    # keep a list to store all prediction dfs
                    all_preds_df_list = []
                    all_metrics_df_list = []
                    for i in range(len(targets)):
                        target = targets[i]

                        predictions, metrics = top_model_voc_predictions(DATA_DIR, PRED_DIR, PRED_SAVE_DIR, info_df_subset,
                                                                         target, libs_list, base_model = base_model, test_seed=seed,
                                                                         model_depth = depth, rank_metric = rank_metric, n_models=n_models)
                        all_preds_df_list.append(predictions)
                        all_metrics_df_list.append(metrics)
                        # extract average prediction scores
                        predictions = predictions[['ID', 'avg_score']]
                        if i == 0:
                            total_voc_df = predictions[['ID']]
                            total_voc_df[target] = predictions['avg_score']
                        else:
                            total_voc_df[target] = predictions['avg_score']
                    # save total_voc_df
                    total_voc_df.to_csv(f"{PRED_SAVE_DIR}/{base_model}seed{seed}_depth{depth}_top{n_models}_{rank_metric}_predicted_all_voc_binding_scores.csv")
                    # filter total_voc_df with canon_df (that's been trimmed down to main canonical sequences)
                    canon_df = pd.read_csv(f"{DATA_DIR}/Canonical_spike_330530.csv", index_col=0)
                    canon_voc_predictions = pd.merge(canon_df, total_voc_df, on=['ID'], how='inner')

                    # merge all predictions into a large df and save
                    all_preds_df = pd.DataFrame()
                    for i in range(len(all_preds_df_list)):
                        main_df = all_preds_df_list[i]
                        main_df = main_df.drop(columns=['avg_score'])
                        if i == 0:
                            all_preds_df = pd.merge(canon_df, main_df, how = 'inner',
                                                    on = ['ID', 'VariantName', 'VariantClass'])
                        else:
                            all_preds_df = pd.merge(all_preds_df, main_df, how = 'inner',
                                                    on = ['ID', 'VariantName', 'VariantClass'])
                    all_preds_df.to_csv(f"{PRED_SAVE_DIR}/{base_model}seed{seed}_depth{depth}_top{n_models}_{rank_metric}_predicted_all_voc_binding_scores_detailed.csv")

                    # merge all metrics into a large df and save
                    all_metrics_df = pd.concat(all_metrics_df_list)
                    all_metrics_df.to_csv(f"{PRED_SAVE_DIR}/{base_model}seed{seed}_depth{depth}_top{n_models}_{rank_metric}_metrics_detailed.csv")

                    # plot heatmap
                    voc_heatmap_df = canon_voc_predictions[['VariantName'] + targets]
                    voc_heatmap_df.set_index('VariantName', inplace=True)
                    # ranked voc by similiarity from (https://covariants.org/shared-mutations)
                    voc_order = ['BA.1', 'BA.1.1', 'BA.3', 'BA.2', 'BA.2.12.1', 'BA.4', 'BQ.1', 'BA.2.75', 'XBB.1']
                    voc_heatmap_df = voc_heatmap_df.reindex(voc_order)
                    sns.heatmap(voc_heatmap_df, cmap='coolwarm')
                    plt.title(f'{base_model}_seed{seed}_Depth{depth}: \n Top {n_models} Models by {rank_metric} predicted binding to VoC')
                    plt.tight_layout()
                    plt.savefig(f"{PRED_SAVE_DIR}/{base_model}_seed{seed}_depth{depth}_top{n_models}_{rank_metric}_predicted_omicron_voc_binding.png", dpi=300, format='png')
                    plt.show()
                    plt.close()
                    # save omicron voc df
                    voc_heatmap_df.to_csv(f"{PRED_SAVE_DIR}/{base_model}_seed{seed}_depth{depth}_top{n_models}_{rank_metric}_predicted_omicron_voc_binding_scores.csv")


