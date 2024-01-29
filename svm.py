#!/cluster/home/zhuyang/miniconda3/envs/omicron/bin/python
import torch
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the saved data
saved_data = torch.load("ESM2_embeddings/VYD224_Lib2_labeled.pt")

labels_numpy = saved_data['labels'].numpy()
embeddings_numpy = saved_data['embeddings'].numpy()
data_df = pd.DataFrame({'labels': labels_numpy, 'embeddings': list(embeddings_numpy)})

# Upsampling the minority class
minority_class = data_df['labels'].value_counts().idxmin()
majority_class = data_df['labels'].value_counts().idxmax()
minority_class_count = data_df['labels'].value_counts().min()

desired_ratio = 0.25  
total_count = len(data_df)
upsample_multiplier = (desired_ratio * total_count) / minority_class_count

majority_data = data_df[data_df['labels'] == majority_class]
minority_data = data_df[data_df['labels'] == minority_class]

minority_upsampled = minority_data.sample(n=int(upsample_multiplier * len(minority_data)), replace=True)
upsampled_data = pd.concat([majority_data, minority_upsampled])
upsampled_data = upsampled_data.sample(frac=1).reset_index(drop=True)

# Extract embeddings and labels
embeddings = np.array(upsampled_data['embeddings'].tolist())
labels = upsampled_data['labels'].values
  
# SVM
accuracies, f1_scores, mcc_scores, micro_f1_scores = [], [], [], []
for seed in range(1, 6):
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=seed, stratify=labels)
    svm_model = make_pipeline(StandardScaler(), SVC(kernel='rbf'))
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    # Calculate meteics
    accuracies.append(accuracy_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))
    mcc_scores.append(matthews_corrcoef(y_test, y_pred))
    micro_f1_scores.append(f1_score(y_test, y_pred, average='micro'))

# Calculate average and standard deviation
avg_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)

avg_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)

avg_mcc = np.mean(mcc_scores)
std_mcc = np.std(mcc_scores)

avg_micro_f1 = np.mean(micro_f1_scores)
std_micro_f1 = np.std(micro_f1_scores)

# Save the metrics
metrics_data = {
    'Average Accuracy': [avg_accuracy],
    'Accuracy SD': [std_accuracy],
    'Average F1 Score': [avg_f1],
    'F1 Score SD': [std_f1],
    'Average MCC': [avg_mcc],
    'MCC SD': [std_mcc],
    'Average Micro F1 Score': [avg_micro_f1],
    'Micro F1 Score SD': [std_micro_f1]
}

results = pd.DataFrame(metrics_data)
results.to_csv('results/esm2/lib2/VYD224.csv', index=False)

print("complete")