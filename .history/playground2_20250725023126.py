import torch
from torch.nn import Sequential, MSELoss, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset, random_split

from condensers import MappingCondenser
from tokenizers import TagValueTokenizer
from layers import AVNNType1Linear, AVNNType2Linear
from blocks import AVNNLinearBlock, AVNNConv2dBlock
from bridges import AVNNLinearToConv2dBridge, AVNNConv2dToLinarBridge

# ----------------------
# 1. Load and tokenize data
# ----------------------
tokenizer = TagValueTokenizer()
tokenizer.exclude_columns([1, 2])  # adjust if needed

# Load your Pokémon tabular data here
with open("data.csv") as f:
    data = f.read()

tensor = tokenizer.tokenize(data)  # shape: [N, D, 2]
tensor = tensor.float()
dim = tensor.shape[1]

# Dummy labels: e.g., total base stat sum (you can replace this)
targets = tensor[:, :, 0].sum(dim=1)  # regression-style label
targets = targets.unsqueeze(-1)  # shape [N, 1]

# ----------------------
# 2. Define AVNN model
# ----------------------
class AVNNPokemonModel(torch.nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim

        # NOTE: linear → F=latent_dim
        self.model = Sequential(
            AVNNType1Linear(input_dim=dim, output_dim=64),
            AVNNType2Linear(input_dim=64, output_dim=32),
            MappingCondenser(),  # or Linear(32, 1) if it’s regressing
        )

        self.output_head = torch.nn.Linear(latent_dim, 1)  # Final activator only

    def forward(self, x):
        x = self.model(x)  # [B, D, 2]
        activator = x[..., 0]  # Use only activator part
        return self.output_head(activator)  # [B, 1]

# ----------------------
# 3. Train
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = TensorDataset(tensor, targets)
train_size = int(len(dataset) * 0.8)
train_data, val_data = random_split(dataset, [train_size, len(dataset) - train_size])
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

model = AVNNPokemonModel(input_dim=dim).to(device)
optimizer = Adam(model.parameters(), lr=1e-3)
criterion = MSELoss()  # or CrossEntropyLoss if you're doing classification

for epoch in range(50):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            print("Target:", batch_y[:5])
            pred = model(batch_x)
            print("Pred: ", pred[0], "Shape: ", pred.shape) # one item
            loss = criterion(pred, batch_y)
            print("Loss: ", loss.item())
            val_loss += loss.item()

    print(f"Epoch {epoch+1} - Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f}")
