#!/cluster/home/zhuyang/miniconda3/envs/omicron/bin/python
import torch
import pandas as pd
import numpy as np
import esm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the CSV data
data = pd.read_csv("data/ZCB11_Lib2_labeled.csv")

filtered_data = data[data["Total_sum"] > 1]

# Extract sequences and labels
sequences = filtered_data['aa'].tolist()
labels = np.array(filtered_data['Label'].tolist()) 

# Load the ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model = model.to(device)

model.eval()

sequence_representations = []

# Batch process sequences to obtain representations
for seq in sequences:
    batch_labels, batch_strs, batch_tokens = batch_converter([(None, seq)])
    with torch.no_grad():
        results = model(batch_tokens.to(device), repr_layers=[33])
    sequence_representations.append(results['representations'][33].mean(1).squeeze().cpu().numpy())

sequences_array = np.array(sequence_representations)
labels_array = labels

# Convert numpy arrays to PyTorch tensors
sequences_tensor = torch.tensor(sequences_array)
labels_tensor = torch.tensor(labels_array)

# Combine tensors into a dictionary
data_to_save = {
    'embeddings': sequences_tensor,
    'labels': labels_tensor
}

combined_file = "ESM2_embeddings/ZCB11_Lib2_labeled.pt"
torch.save(data_to_save, combined_file)

print("completed")