"""
Author: Andrey Ignatov
Email: andrey.ignatoff@gmail.com

This script provides an implementation of baseline models (XGBoost, SVM, Random Forest, Logistic Regression, Naive Bayes,
 KNN) trained for binding / antibody escape prediction from sequence data.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from baseline_models_utils import *
import sys

np.random.seed(42)

# Specify the target, available options:
# 2_7, A23581, ACE2, ADG20, BRII198, Cov22130, Cov22196, S2H97, S2X259, VYD222, VYD223, VYD224, ZCB11
target = "2_7"

# Specify the lib, available options:
# Lib1 , Lib2
lib = "Lib1"

# Specify the path to the folder containing your training data
path_to_dataset = "data/"

# Specify the machine learning model to be trained, available options:
# XGBoost, SVM, RandomForest, Logistic, NaiveBayes, KNN
model_option = "NaiveBayes"

# Specify data filtering threshold based on raw sequence counts
# 1 means no filtering is applied
threshold_count_train = 1
threshold_count_test = 2

# Specify the number of folds in cross-validation
k_folds = 5


if __name__ == "__main__":

    print("Loading data...")

    sequences, labels, counts = load_data(target=target, lib=lib, path_to_dataset=path_to_dataset)

    print("Encoding sequences...")

    if model_option == "RandomForest" or model_option == "NaiveBayes":
        data = encode_numerical(sequences)
    else:
        data = encode_onehot_padded(sequences, flatten=True)

    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    mcc_score_list = []

    score_table = np.zeros((k_folds, 5))

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=4)

    print("Training model...")

    for i, (train_index, test_index) in enumerate(kf.split(data)):

            print("Cross-Validation, Fold %d / %d" % (i+1, k_folds))

            X_train = data[train_index]
            Y_train = labels[train_index]
            count_train = counts[train_index]

            X_test = data[test_index]
            Y_test = labels[test_index]
            count_test = counts[test_index]

            X_train, Y_train = filter_dataset(X_train, Y_train, count_train, threshold_count_train)
            X_test, Y_test = filter_dataset(X_test, Y_test, count_test, threshold_count_test)

            model = None

            if model_option == "Logistic":
                model = LogisticRegression(C=50, class_weight="balanced", fit_intercept=True)

            if model_option == "RandomForest":
                model = RandomForestClassifier(n_estimators=500, n_jobs=8, class_weight="balanced")

            if model_option == "NaiveBayes":
                model = CategoricalNB(alpha=1.0, min_categories=25, fit_prior=False)

            if model_option == "SVM":
                model = LinearSVC(C=1, class_weight="balanced")

            if model_option == "KNN":
                model = KNeighborsClassifier(n_neighbors=9, metric='cosine', weights='distance', n_jobs=8)

            if model_option == "XGBoost":
                model = XGBClassifier(n_estimators=500, max_depth=10, learning_rate=1, objective='binary:logistic',
                                      scale_pos_weight=np.sum(Y_train == 0) / np.sum(Y_train == 1))

            if model is None:
                print("Please specify the correct model.")
                print("Available model list: RandomForest, Logistic, XGBoost, SVM, KNN, NaiveBayes")
                sys.exit()

            # Training model

            model.fit(X_train, Y_train)

            # Generating predictions

            predictions = model.predict(X_test)

            # Computing accuracy metrics

            score_table[i] = compute_accuracy_scores(Y_test, predictions)

            print("Accuracy: %.2f, MCC Score: %.4f" % (score_table[i][0], score_table[i][4]))

    print("---------------------------------------------------")
    print("Target '%s', library '%s', average results:" % (target, lib))
    print("---------------------------------------------------")

    scores_avg = np.mean(score_table, axis=0)
    scores_std = np.std(score_table, axis=0)

    print("Accuracy:\t%.2f ± %.2f" % (scores_avg[0], scores_std[0]))
    print("Precision:\t%.2f ± %.2f" % (scores_avg[1], scores_std[1]))
    print("Recall:\t\t%.2f ± %.2f" % (scores_avg[2], scores_std[2]))
    print("F1 Score:\t%.4f ± %.4f" % (scores_avg[3], scores_std[3]))
    print("MCC:\t\t%.4f ± %.4f" % (scores_avg[4], scores_std[4]))
