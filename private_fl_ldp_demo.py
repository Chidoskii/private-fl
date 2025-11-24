import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus import PrivacyEngine
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------
# BASE URL OF DATA HOST LOCATION
# ----------------------------------------------------
BASE_URL = "https://raw.githubusercontent.com/Chidoskii/private-fl/main/health_data_"


# --------------------------------------
# MODEL DEFINITION
# --------------------------------------
class ExampleLogisticModule(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        out = torch.sigmoid(self.linear(x))
        return out[:, 0]


# ---------------------------
# LOAD REMOTE CSV AS A DATASET
# ---------------------------
def load_dataset_from_url(url):
    df = pd.read_csv(url)

    # Input features (modify if needed)
    X = df[["age", "sex", "blood", "admission"]].values.astype(float)

    # Binary label
    y = df["result"].values.astype(float)

    tensor_x = torch.tensor(X).float()
    tensor_y = torch.tensor(y).float()

    return TensorDataset(tensor_x, tensor_y)


# ---------------------------
# CREATE 10 CLIENT DATASETS
# ---------------------------
client_datasets = []

for i in range(1, 11):
    url = f"{BASE_URL}{i}.csv"
    print("Loading:", url)
    dataset = load_dataset_from_url(url)
    client_datasets.append(dataset)


# ---------------------------
# FEDERATED SETTINGS
# ---------------------------
NUM_CLIENTS = 10
LOCAL_EPOCHS = 3
ROUNDS = 5
BATCH_SIZE = 64
LR = 0.01
input_size = 4  # age, sex, blood, admission


# ---------------------------
# FEDERATED AVERAGING FUNCTION
# ---------------------------
def fed_avg(models):
    """
    Federated averaging over a list of client models.
    Only averages parameters that are present in ALL models.
    """
    # Collect state dicts from all client models
    state_dicts = [m.state_dict() for m in models]

    # Compute intersection of keys across all models
    common_keys = set(state_dicts[0].keys())
    for sd in state_dicts[1:]:
        common_keys &= set(sd.keys())

    if not common_keys:
        raise ValueError("No common parameter keys found across client models!")

    # (Optional) debug: see what keys we are averaging
    print("Common parameter keys being averaged:", common_keys)

    # Create a fresh global model
    global_model = ExampleLogisticModule(input_size)
    global_state = global_model.state_dict()

    # Average only over common keys
    for key in common_keys:
        # Stack the tensors for this key across all models and average
        stacked = torch.stack([sd[key].float() for sd in state_dicts], dim=0)
        global_state[key] = stacked.mean(dim=0)

    # Load averaged weights; keep other params as originally initialized
    global_model.load_state_dict(global_state, strict=False)

    return global_model


# ---------------------------
# LOCAL CLIENT TRAINING (DP)
# ---------------------------
def train_local_model(dataset, seed):
    torch.manual_seed(seed)

    model = ExampleLogisticModule(input_size)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    privacy_engine = PrivacyEngine()
    model, optimizer, loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=1.0,
        max_grad_norm=0.5,
    )

    for epoch in range(LOCAL_EPOCHS):
        for inputs, targets in loader:
            outputs = model(inputs)
            loss = F.binary_cross_entropy(outputs, targets)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

    eps = privacy_engine.get_epsilon(delta=1e-6)
    return model, eps


# ---------------------------
# FEDERATED TRAINING LOOP
# ---------------------------
global_model = ExampleLogisticModule(input_size)
privacy_eps = []

for round in range(ROUNDS):
    print(f"\n--- Federated Round {round+1} ---")
    local_models = []

    for client_id in range(NUM_CLIENTS):
        model, eps = train_local_model(client_datasets[client_id], seed=client_id)
        local_models.append(model)
        privacy_eps.append(eps)
        print(f" Client {client_id+1} DP-epsilon = {eps:.2f}")

    global_model = fed_avg(local_models)

print("\nFederated DP Training Complete.")
print("Final global model state dict:")
print(global_model.state_dict())
