#!/cluster/home/zhuyang/miniconda3/envs/omicron/bin/python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import esm
import pandas as pd
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the CSV data
data = pd.read_csv("data/S2H97_Lib1_labeled.csv")
filtered_data = data[data["Total_sum"] > 1]

# Extract amino acid sequences and labels
sequences2 = filtered_data['aa'].tolist()
labels2 = np.array(filtered_data['Label'].tolist())

# Upsampling the minority class
minority_class = filtered_data['Label'].value_counts().idxmin()
majority_class = filtered_data['Label'].value_counts().idxmax()
minority_class_count = filtered_data['Label'].value_counts().min()

desired_ratio = 0.25  
total_count = len(filtered_data)
upsample_multiplier = (desired_ratio * total_count) / minority_class_count

majority_data = filtered_data[filtered_data['Label'] == majority_class]
minority_data = filtered_data[filtered_data['Label'] == minority_class]

minority_upsampled = minority_data.sample(n=int(upsample_multiplier * len(minority_data)), replace=True)
upsampled_data = pd.concat([majority_data, minority_upsampled])
upsampled_data = upsampled_data.sample(frac=1).reset_index(drop=True)

# Extract amino acid sequences and labels
sequences = upsampled_data['aa'].tolist()
labels = np.array(upsampled_data['Label'].tolist())

# Load the ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model = model.to(device) 

# Freeze all layers 
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last layer
for name, param in model.named_parameters(): 
    if "layers.32" in name or "lm_head" in name or "contact_head" in name or "emb_layer_norm_after" in name :
        param.requires_grad = True

model.train()

# Define the ESM2Predictor class
class ESM2Predictor(nn.Module):
    def __init__(self, esm_model):
        super().__init__()
        self.encoder = esm_model
        self.predictor = nn.Sequential(
            nn.Linear(1280, 2),
            nn.LogSoftmax(dim=1) 
        )

    def forward(self, seq, precomputed=False):
        if precomputed:
            rep = seq
        else:
            res = self.encoder(seq, repr_layers=[33], return_contacts=False)
            rep = res['representations'][33]
            rep = rep.mean(1)
        y = self.predictor(rep)
        return y

predictor_model = ESM2Predictor(model).to(device)

initial_batch_size = 8
num_batches = len(sequences) // initial_batch_size + (len(sequences) % initial_batch_size > 0)

sequence_representations = []
for b in range(num_batches):
    start_idx = b * initial_batch_size
    end_idx = (b + 1) * initial_batch_size
    data_batch = [(f"protein_{i+start_idx}", seq) for i, seq in enumerate(sequences[start_idx:end_idx])]
    batch_labels, batch_strs, batch_tokens = batch_converter(data_batch)

    with torch.no_grad():
        results = model(batch_tokens.to(device).long(), repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33].cpu()  

    for i, tokens_len in enumerate((batch_tokens != alphabet.padding_idx).sum(1)):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0).numpy())

# Convert sequence representations and labels to tensors
sequences_tensor = torch.tensor(np.array(sequence_representations), dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.long)  

# Split into test, val, train sets
test_size = 0.1
val_size = 0.1 
train_size = 1 - test_size - val_size 
val_size_corrected = val_size / (train_size + val_size) 

train_val_data, test_data, train_val_labels, test_labels = train_test_split(
    sequences_tensor, labels_tensor, test_size=test_size, random_state=11, stratify=labels_tensor)
testsize = len(test_data)

train_data, val_data, train_labels, val_labels = train_test_split(
    train_val_data, train_val_labels, test_size=val_size_corrected, random_state=11, stratify=train_val_labels)
trainsize = len(train_data)

# Create DataLoader objects
batch_size = 32  
train_dataset = TensorDataset(train_data, train_labels)
val_dataset = TensorDataset(val_data, val_labels)
test_dataset = TensorDataset(test_data, test_labels)  
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  

# Set up optimizer, loss function, and learning rate scheduler
optimizer = optim.Adam(filter(lambda p: p.requires_grad, predictor_model.parameters()), lr=1e-3, weight_decay=1e-5)
loss_fn = nn.NLLLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.05)  

# Training loop
n_epochs = 50
for epoch in range(n_epochs):
    predictor_model.train()
    train_loss = 0.0

    for batch_data, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = predictor_model(batch_data.to(device), precomputed=True)
        loss = loss_fn(outputs, batch_labels.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor_model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()

    scheduler.step()

    # Validation
    predictor_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_data, val_labels in val_loader:
            val_outputs = predictor_model(val_data.to(device), precomputed=True)
            v_loss = loss_fn(val_outputs, val_labels.to(device))
            val_loss += v_loss.item()

    print(f'Epoch {epoch+1}/{n_epochs}, Training Loss: {train_loss/len(train_loader)}, Validation Loss: {val_loss/len(val_loader)}')

# Get finetuned embeddings
fine_tuned_embeddings = []

initial_batch_size = 8
num_batches = len(sequences2) // initial_batch_size + (len(sequences2) % initial_batch_size > 0)

for b in range(num_batches):
    start_idx = b * initial_batch_size
    end_idx = (b + 1) * initial_batch_size
    data_batch = [(f"protein_{i+start_idx}", seq) for i, seq in enumerate(sequences2[start_idx:end_idx])]
    batch_labels, batch_strs, batch_tokens = batch_converter(data_batch)

    with torch.no_grad():
        results = model(batch_tokens.to(device), repr_layers=[33])
    representations = results["representations"][33].cpu()

    for i, tokens_len in enumerate((batch_tokens != alphabet.padding_idx).sum(1)):
        fine_tuned_embeddings.append(representations[i, 1:tokens_len-1].mean(0).numpy())

sequences_array = np.array(fine_tuned_embeddings)
labels_array = labels2

# Convert numpy arrays to PyTorch tensors
sequences_tensor = torch.tensor(sequences_array)
labels_tensor = torch.tensor(labels_array)

# Combine tensors into a dictionary
data_to_save = {
    'embeddings': sequences_tensor,
    'labels': labels_tensor
}

combined_file = "finetune_embeddings/S2H97_Lib1_labeled.pt"
torch.save(data_to_save, combined_file)