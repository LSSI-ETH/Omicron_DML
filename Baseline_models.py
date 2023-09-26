"""
Authors: Beichen Gao, Benedikt Eisinger, Rachita Kumar

This script was made for fitting and evaluating different baseline models using random search CV hyperparameter optimization.
All functions use relative paths to save outputs because the main function that is called creates a directory
structure and switches directories.

Before starting you need to change:
    WORK_DIR, SAVE_DIR, LIB_DIR
    WORK_DIR should contain helpers.py

To run the script you have to:
    - Define a list object with the library numbers e.g. ['Lib1', 'Lib2', ...]
    - Define a list object with the model names as used fo the input 'model_type' of the evaluate_models() function.
        E.g. ['SGD', 'RF', 'RBF'].
    - Decide about how many rounds of ext CV you want.

The script should be easily expandable by adding a function implementing RandomizedSearchCV() for the new model.
Adding the evaluation to the evaluate_models() function (copy paste code block and change names and define new
hyperparameters). And adding the model type to the correct block of the evaluate_scores() function.
"""

import os

# set directories if working locally. File path should have a '/' at the end!!!!
WORK_DIR = os.getcwd()
# LIB_DIR contains the labeled libraries
LIB_DIR = f"{WORK_DIR}/data/"
# SAVE_DIR is the location for all output
SAVE_DIR = f"{WORK_DIR}/baseline_models/"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

rand_state = 42

import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    f1_score,
    roc_curve,
    auc,
    make_scorer,
)

import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from helpers import encode_onehot_padded, flatten_matrix, levenshtein_RBD


def encode_data(df):
    """
    Encodes the data.
    Input:
        df: dataframe containing labeled sequences
    """
    x = flatten_matrix(encode_onehot_padded(df.aa))
    y = df.Label
    return x, y


def evaluate_scores(df, model, model_type):
    """
    Evaluates the model and returns various metrics to estimate the models performance.
    Some of the options given are not used in the code.
    Input:
        df: dataframe containing labeled sequences
        model: The model
        model_type: The type of model. Here is a bit unnecessary because we only have one but if multiple models
        are implemented in one script it would be needed.
    Output: List element with 4 values and 2 individual values.
    """
    if (
            model_type == "Log_reg"
            or model_type == "NB"
            or model_type == "KN"
            or model_type == "RF"
    ):
        x = flatten_matrix(encode_onehot_padded(df.aa))
        y_pred = model.predict(x)
        y_pred_prob = model.predict_proba(x)[:, 1].ravel()
    elif model_type == "SVM":
        x = flatten_matrix(encode_onehot_padded(df.aa))
        y_pred = model.predict(x)
        # For now I will just initialize a value here so that the code can hopefully run
        y_pred_prob = model.predict(x)
        # y_pred_prob = model.predict(x)
    elif model_type == "SGD":
        loss = model["sgdclassifier"]
        if (
                str(loss) != "SGDClassifier(loss='modified_huber')"
                or str(loss) != "SGDClassifier(loss='log_loss')"
        ):
            x = flatten_matrix(encode_onehot_padded(df.aa))
            y_pred = model.predict(x)
            # For now I will just initialize a value here so that the code can hopefully run
            y_pred_prob = model.predict(x)
            # y_pred_prob = model.predict(x)
        else:
            x = flatten_matrix(encode_onehot_padded(df.aa))
            y_pred = model.predict(x)
            y_pred_prob = model.predict_proba(x)[:, 1].ravel()
    y_true = df.Label
    # Calculate scores
    acc = accuracy_score(y_true, y_pred)
    # acc = acc.astype(np.float)
    f1 = f1_score(y_true, y_pred)
    # f1 = f1.astype(np.float)
    mcc = matthews_corrcoef(y_true, y_pred)
    # mcc = mcc.astype(np.float)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    model_auc = auc(fpr, tpr)
    # model_auc = model_auc.astype(np.float)
    fpr = fpr.astype(np.float)
    tpr = tpr.astype(np.float)
    return [acc, f1, mcc, model_auc], fpr, tpr


def make_balanced_dataset(data):
    """
    Balances out the dataset. This function was specifically made for datasets in which there are more sequences with
    negative (0) label. It returns a balanced dataset and the remaining negative data which is later concatenated to
    the test set.
    Input:
        data: Dataframe containing labeled sequences
    Output:
        new_dataset: The balanced dataset which is going to be split into train and test data.
        neg_remain: The remaining negative values which are going to be added to the test data.
    """
    pos = data.loc[data["Label"] == 1]
    neg = data.loc[data["Label"] == 0]
    neg_sub = neg.sample(n=len(pos))
    neg_remain = neg.drop(neg_sub.index, axis=0).reset_index(drop=True)
    new_dataset = pd.concat([pos, neg_sub]).reset_index(drop=True)
    return new_dataset, neg_remain


def calculate_distance(df, name, RBD_seq, max_dist=16):
    """
    Parameters
    ----------
    df : Pandas Dataframe
        Imported Pandas Dataframe read in from .tsv file with AA sequence in "junction_aa" column
        and consensus_count as the count column.
    max_dist : Int
        Maximum LD distance expected (default is 16)
    name : String
        Name of the dataframe for saving processed data into a separate file and for the graph.

    Returns
    -------
    Cleaned dataframe, barplot of distances, and saves a copy of the cleaned dataframe as .tsv.

    Requires
    -------
    levenshtein_RBD function, make_dist_df function, matplotlib
    """
    # Add Distance to cleaned and original df for comparison
    df["Distance"] = df.apply(lambda row: levenshtein_RBD(row.aa, RBD_seq), axis=1)
    # Add 'Target' column
    df["Target"] = name
    # Keep only those with distance below max_dist threshold
    df = df[df.Distance <= max_dist]
    return df


def anti_merge(df_a, df_b):
    merge_df = pd.merge(df_a, df_b, how="outer", on=["aa", "Label"], indicator=True)
    merge_df_left = merge_df[merge_df["_merge"] == "left_only"]
    merge_df_left.drop(["Distance_y", "_merge"], axis=1, inplace=True)
    merge_df_left.rename(columns={"Distance_x": "Distance"}, inplace=True)
    return merge_df_left


def make_dist_df(df):
    """
    Takes input df of annotated sequences, and creates a dataframe of sequence counts by distance
    """
    c = df.Distance.value_counts()
    c = pd.DataFrame(c).reset_index()
    c = c.rename(columns={"index": "Distance", "Distance": "Counts"})
    c.sort_values("Distance", inplace=True)
    return c


def balance_edit_distance(df, distances, name, rand_state):
    dist = distances
    balanced_list = pd.DataFrame()
    for i in dist:
        pos = df[(df["Label"] == 1)]
        neg = df[(df["Label"] == 0)]
        a = pos.loc[pos["Distance"] == i]
        b = neg.loc[neg["Distance"] == i]
        if len(a) <= len(b):
            balanced_list = balanced_list.append(a)
            balanced_list = balanced_list.append(
                b.sample(len(a), random_state=rand_state)
            )
        if len(b) < len(a):
            balanced_list = balanced_list.append(
                a.sample(len(b), random_state=rand_state)
            )
            balanced_list = balanced_list.append(b)

    minimum = int(min(balanced_list.Distance))
    maximum = int(max(balanced_list.Distance))
    graph_labels = list(map(str, range(minimum, (maximum + 1))))
    graph_labels_df = pd.DataFrame(data=graph_labels, columns={"Distance"})
    graph_labels_df["Distance"] = graph_labels_df["Distance"].astype(
        int
    )  # Need to reset type to merge

    # Make positive and negative sets by label
    pos_set = make_dist_df(balanced_list[balanced_list.Label == 1])
    pos_set["Distance"] = pos_set["Distance"].astype(int)

    neg_set = make_dist_df(balanced_list[balanced_list.Label == 0])
    neg_set["Distance"] = neg_set["Distance"].astype(int)

    # Merge with graph_labels_df for final full dfs for graphing
    graph_pos = pd.merge(graph_labels_df, pos_set, how="left", on="Distance")
    graph_pos.fillna(0, inplace=True)

    graph_neg = pd.merge(graph_labels_df, neg_set, how="left", on="Distance")
    graph_neg.fillna(0, inplace=True)

    # Now prepare for graph
    pos_counts = graph_pos["Counts"].squeeze()
    neg_counts = graph_neg["Counts"].squeeze()

    a = np.arange(len(graph_labels))
    width = 0.35

    fig, ax = plt.subplots()

    rects1 = ax.bar(a - width / 2, pos_counts, width, label="Positive", color="orange")
    rects2 = ax.bar(a + width / 2, neg_counts, width, label="Negative", color="blue")

    ax.set_ylabel("Counts")
    ax.set_title(name + " Balanced Positive and Negative Sequence Counts by Distance")
    ax.set_xticks(a)
    ax.set_xticklabels(graph_labels)
    ax.legend()

    fig.tight_layout()
    plt.show()
    plt.savefig(f'{name}_balanced_sequence_distribution.png', dpi=300, format='png')
    plt.close()

    # Add in a section where we keep the other sequences left out
    unseen_df = anti_merge(df, balanced_list)

    # balanced_list.to_csv(name + '_balanced.csv', index=False)
    # unseen_df.to_csv(name + '_remaining.csv', index=False)
    return balanced_list, unseen_df


def make_ROC_plot(fpr, tpr, num_round, model_type, scores):
    """
    This function takes the output from evaluate_scores() and makes ROC plots from it.
    Input:
        fpr, tpr: Output of evaluate_scores()
        num_round: The number the external CV round
        model_type: The type of model
            ## Both of the above are needed for naming the output file ##
        scores = Output of evaluate_scores()
    Output: ROC plot
    """
    fpr_tpr_df = pd.DataFrame()
    fpr_tpr_df["fpr"] = fpr.tolist()
    fpr_tpr_df["tpr"] = tpr.tolist()
    fpr_tpr_df.to_csv(str(model_type) + "_fpr_tpr_df_" + num_round + ".csv")
    scores_df = pd.DataFrame()
    scores_df["vals"] = scores
    # Making roc curve
    plt.figure()
    lw = 2
    plt.plot(
        fpr_tpr_df["fpr"],
        fpr_tpr_df["tpr"],
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % float(scores_df["vals"][3]),
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve for " + str(model_type) + " in round " + num_round)
    plt.legend(loc="lower right")
    plt.savefig(f'{model_type}_ROC_curve_{num_round}.png', dpi=300, format='png')
    plt.show()
    plt.close()


def save_round_scores(scores, model_type, num_rounds):
    """
    Saves round scores (external CV scores) made with the evaluate_model() function.
    Function is designed to automatically determine the number of inputs it gets. For that to work, the scores must be
    saved per model_type. For plotting the values are later concatenated in the make_scores_plot() function.
    Input:
        scores: Output of evaluate_model()
        model_type, num_rounds: Model type and number of external CV round needed for saving to a new file.
    """
    df = pd.DataFrame(scores)
    indices = []
    for i in range(1, num_rounds + 1):
        new_index = str(model_type) + "_" + str(i)
        indices.append(new_index)
    df.index = indices
    df.columns = ["acc", "f1", "mcc", "model_auc"]
    df.to_csv("ext_round_scores.csv")


def collect_counts(balanced_df, train_df, test_df_full):
    """
    This function collects counts and returns them as a single object so that we can save them later on with
    the save_count_csv() function.
    Inputs:
        - balanced_df: Dataframe that is created by the make_balanced_dataset function which is needed to make a
        training dataset that contains the same number of positive and negative labels
        - train_df: The training dataset which is produced by the train_test_split function from sklearn
        - test_df_full: The FULL test dataset, which is produced by adding the removed values (removed to make balanced
        dataset).
    """
    # Isolate total length
    n_bal = len(balanced_df.Label)
    n_train = len(train_df.Label)
    n_test = len(test_df_full.Label)

    # Isolate n-seqs with positive label
    n_bal_pos = balanced_df.Label.value_counts()[1]
    n_train_pos = train_df.Label.value_counts()[1]
    n_test_pos = test_df_full.Label.value_counts()[1]

    # Isolate n-seqs with negative label
    n_bal_neg = balanced_df.Label.value_counts()[0]
    n_train_neg = train_df.Label.value_counts()[0]
    n_test_neg = test_df_full.Label.value_counts()[0]

    return [
        n_bal,
        n_train,
        n_test,
        n_bal_pos,
        n_train_pos,
        n_test_pos,
        n_bal_neg,
        n_train_neg,
        n_test_neg,
    ]


def save_counts_csv(all_counts, num_rounds):
    """
    Saves the counts into a csv file. Each row is external CV round.
    Input:
        all_counts: Concatenated output of collect_counts()
        num_rounds: The total number of rounds (needed to determine how many rows the df has)
    """
    df = pd.DataFrame(all_counts)
    df.index = pd.RangeIndex(start=1, stop=(num_rounds + 1), step=1)
    df.columns = [
        "n_bal",
        "n_train",
        "n_test",
        "n_bal_pos",
        "n_train_pos",
        "n_test_pos",
        "n_bal_neg",
        "n_train_neg",
        "n_test_neg",
    ]
    df.to_csv("counts.csv")


def make_scores_plot(path_to_parent_dir, model_list, lib_num):
    """
    Makes a plot in which the three scores (acc, f1, mcc) are plotted per number of neurons on the dense layers and
    library in one plot.
    Input:
        path_to_parent_dir: Path to the parent directory in which the output folder structure is established.
        neurons_list: List of neurons in the dense layers. Needed, becasue we have one output folder per this metric
        e.g. 'neuron_32'
        lib_num: Number of library e.g. 'Lib1'.
    """
    df = pd.DataFrame()
    for model in model_list:
        print(model)
        new_df = pd.read_csv(f"{path_to_parent_dir}/{lib_num}/{model}/ext_round_scores.csv")
        new_df = new_df.drop(["Unnamed: 0", "model_auc"], axis=1).stack().reset_index()
        new_df["model"] = model
        df = pd.concat([df, new_df], axis=0)
        # df = df.reset_index()

    sns.set_palette(sns.color_palette(["#33BBEE", "#009988", "#0077BB"]))
    ax = sns.boxplot(data=df, x="level_1", y=0, hue="model")
    ax.set(xlabel="Metric", ylabel="Score")
    plt.xticks(size=10)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], size=10)
    plt.title("Metric scores for " + str(lib_num))
    plt.savefig(f'{path_to_parent_dir}/{lib_num}/ext_CV_scores_combined.png', dpi=300, format="png")
    plt.show()
    plt.close()


"""
SGD
For explanation of input see above.
"""


def SGD_Random_Search(X_train, y_train, parameters, n_iter=30):
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import RandomizedSearchCV

    clf = make_pipeline(StandardScaler(), SGDClassifier())
    clf_search = RandomizedSearchCV(
        clf,
        param_distributions=parameters,
        n_iter=n_iter,
        cv=5,
        verbose=2,
        scoring="precision",
        return_train_score=True,
        n_jobs=-1,
    )
    clf_search.fit(X_train, y_train)
    best_parameters = clf_search.best_params_
    best_precision = clf_search.best_score_
    results = clf_search.cv_results_
    best_clf = clf_search.best_estimator_
    return best_clf


"""
RF
For explanation of input see above.
"""


def RF_Random_Search(X_train, y_train, parameters, n_iter=30):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV

    clf = RandomForestClassifier()
    clf_search = RandomizedSearchCV(
        clf,
        param_distributions=parameters,
        n_iter=n_iter,
        cv=5,
        verbose=2,
        scoring="precision",
        return_train_score=True,
        n_jobs=-1,
    )
    clf_search.fit(X_train, y_train)
    best_parameters = clf_search.best_params_
    best_precision = clf_search.best_score_
    results = clf_search.cv_results_
    best_clf = clf_search.best_estimator_
    return best_clf


"""
RBF SVM
"""


def RBF_Random_Search(X_train, y_train, parameters, n_iter=30):
    from sklearn.svm import SVC
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import RandomizedSearchCV

    # Could be that we need to make a pipeline here or so
    # clf = LinearSVC()
    clf = make_pipeline(StandardScaler(), SVC())
    clf_search = RandomizedSearchCV(
        clf,
        param_distributions=parameters,
        n_iter=n_iter,
        cv=5,
        verbose=2,
        scoring="precision",
        return_train_score=True,
        n_jobs=-1,
    )
    clf_search.fit(X_train, y_train)
    best_parameters = clf_search.best_params_
    best_precision = clf_search.best_score_
    results = clf_search.cv_results_
    best_clf = clf_search.best_estimator_
    return best_clf


"""
NB
"""


def NB(X_train, y_train):
    from sklearn.naive_bayes import GaussianNB
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(StandardScaler(), GaussianNB())
    clf.fit(X_train, y_train)
    return clf


"""
Log_reg
"""


def Log_reg(X_train, y_train):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(StandardScaler(), LogisticRegression())
    clf.fit(X_train, y_train)
    return clf


"""
Evaluate the models
"""


def evaluate_models(x_train, y_train, test_df_new, model_type, rand_state, rounds):
    if model_type == "SGD":
        SGD_parameters = {
            "sgdclassifier__loss": [
                "hinge",
                "log_loss",
                "log",
                "modified_huber",
                "squared_hinge",
                "perceptron",
                "squared_error",
                "huber",
                "epsilon_insensitive",
                "squared_epsilon_insensitive",
            ],
            "sgdclassifier__penalty": ["l2", "l1", "elasticnet"],
            "sgdclassifier__alpha": [0.0001, 0.001, 0.01],
        }
        print(f"Random State for SGD in round{rounds} is: {rand_state}")
        SGD_model = SGD_Random_Search(x_train, y_train, SGD_parameters)
        model_scores, SGD_fpr, SGD_tpr = evaluate_scores(
            test_df_new, SGD_model, model_type="SGD"
        )
        make_ROC_plot(SGD_fpr, SGD_tpr, rounds, model_type="SGD", scores=model_scores)
        filename = f"SGD_model_round_{rounds}.sav"
        joblib.dump(SGD_model, filename)
    elif model_type == "RF":
        RF_parameters = {
            "n_estimators": [100, 250, 500],
            "criterion": ["gini", "entropy", "log_loss"],
            "max_depth": [20, 40, 60, 80, 100, None],
        }
        print(f"Random State for RF in round{rounds} is: {rand_state}")
        RF_model = RF_Random_Search(x_train, y_train, RF_parameters)
        model_scores, RF_fpr, RF_tpr = evaluate_scores(
            test_df_new, RF_model, model_type="RF"
        )
        make_ROC_plot(RF_fpr, RF_tpr, rounds, model_type="RF", scores=model_scores)
        filename = f"RF_model_round_{rounds}.sav"
        joblib.dump(RF_model, filename)
    elif model_type == "RBF":
        RBF_parameters = {
            "svc__kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
            "svc__degree": [3, 4, 5],
            "svc__C": [1.0, 10, 100],
        }
        print(f"Random State for RBF in round{rounds} is: {rand_state}")
        RBF_model = RBF_Random_Search(x_train, y_train, RBF_parameters)
        model_scores, RBF_fpr, RBF_tpr = evaluate_scores(
            test_df_new, RBF_model, model_type="SVM"
        )
        make_ROC_plot(RBF_fpr, RBF_tpr, rounds, model_type="RBF", scores=model_scores)
        filename = f"RBF_model_round_{rounds}.sav"
        joblib.dump(RBF_model, filename)
    elif model_type == "NB":
        NB_model = NB(x_train, y_train)
        model_scores, NB_fpr, NB_tpr = evaluate_scores(
            test_df_new, NB_model, model_type="NB"
        )
        make_ROC_plot(NB_fpr, NB_tpr, rounds, model_type="NB", scores=model_scores)
        filename = f"NB_model_round_{rounds}.sav"
        joblib.dump(NB_model, filename)
    elif model_type == "Log_reg":
        Log_reg_model = Log_reg(x_train, y_train)
        model_scores, Log_reg_fpr, Log_reg_tpr = evaluate_scores(
            test_df_new, Log_reg_model, model_type="Log_reg"
        )
        make_ROC_plot(
            Log_reg_fpr, Log_reg_tpr, rounds, model_type="Log_reg", scores=model_scores
        )
        filename = f"Log_reg_model_round_{rounds}.sav"
        joblib.dump(Log_reg_model, filename)
    return model_scores


def loop_score_models(
        lib_dir, num_rounds, save_dir, target_list, lib_list, model_list, wt_seq
):
    # Save old workdir
    old_dir = os.getcwd()

    for tar in target_list:
        for lib in lib_list:
            seq_df = pd.read_csv(f"{lib_dir}/{tar}_{lib}_labeled.csv")
            # Calculate distances
            seq_df = calculate_distance(seq_df, tar, wt_seq, max_dist=16)
            for model in model_list:
                # Create the output folder structure needed to run the code
                import pathlib

                new_dir = f"{save_dir}/{tar}/{lib}/{model}/"
                pathlib.Path(new_dir).mkdir(parents=True, exist_ok=True)

                # Set new workdir for saving
                print(f"Saving output into: {new_dir}")
                os.chdir(new_dir)

                # Make objects to collect counts and scores
                all_counts = []
                round_scores = []

                # Determine of how many rounds to do external CV for from user input
                rounds_list = list(map(str, range(1, num_rounds + 1)))
                for rounds in rounds_list:
                    # Set the random state
                    rand_state = random.randint(1, 420)
                    distances = list(range(1, 15))
                    # balanced_df, neg_remain = make_balanced_dataset(seq_df)
                    balanced_df, neg_remain = balance_edit_distance(
                        seq_df, distances, tar, rand_state
                    )
                    balanced_df.to_csv(f"Balanced_df_{rounds}.csv")
                    neg_remain.to_csv(f"Remaining_neg_{rounds}.csv")

                    # Train/test split and save the dfs
                    train_df, test_df = train_test_split(
                        balanced_df, test_size=0.2, random_state=rand_state
                    )
                    train_df.to_csv(f"Train_df_{rounds}.csv")
                    test_df.to_csv(f"Test_df_before_imbalance_{rounds}.csv")
                    test_df_new = pd.concat([test_df, neg_remain]).reset_index(
                        drop=True
                    )
                    test_df_new.to_csv(f"Test_df_after_imbalance_{rounds}.csv")

                    # Collect statistics on the datasets and append them to one list that is later saved
                    # (per lib and per amount of nodes, there will be one list containing all values from all rounds)
                    counts = collect_counts(balanced_df, train_df, test_df_new)
                    all_counts.append(counts)

                    # Encode data. the function is copied and Log_reg is defined as model type because it comes from and
                    # older script
                    x_train, y_train = encode_data(train_df)

                    model_scores = evaluate_models(
                        x_train, y_train, test_df_new, model, rand_state, rounds
                    )
                    round_scores.append(model_scores)

                save_counts_csv(all_counts, len(rounds_list))
                save_round_scores(round_scores, model_type=model, num_rounds=num_rounds)

                # Reset into previous workdir
                os.chdir(old_dir)

            # The plot will be saved into the lib directory.
            make_scores_plot(save_dir, model_list, lib)


os.chdir(WORK_DIR)
target_list = ["ACE2"]
lib_list = ["Lib1", "Lib2"]
model_list = ["SGD", "RF", "RBF", "NB", "Log_reg"]
wt_seq = (
    "NITNLCPFDEVFNATRFASVYAWNRKRISNCVADYSVLYNLAPFFTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQT"
    "GNIADYNYKLPDDFTGCVIAWNSNKLDSKVSGNYNYLYRLFRKSNLKPFERDISTEIYQAGNKPCNGVAGFNCYFPLRSYSFRPTYG"
    "VGHQPYRVVVLSFELLHAPATVCGPKKST"
)

loop_score_models(LIB_DIR, 5, SAVE_DIR, target_list, lib_list, model_list, wt_seq)
