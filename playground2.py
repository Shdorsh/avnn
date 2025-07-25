import torch
from torch.nn import Module, MSELoss, Sequential, Linear
from torch.utils.data import DataLoader, random_split, TensorDataset
from torch.optim import Adam

# === Custom AVNN imports ===
from tokenizers import TagValueTokenizer
from layers import AVNNType1Linear, AVNNType2Linear
from blocks import AVNNLinearBlock, AVNNConv2dBlock
from bridges import AVNNLinearToConv2dBridge, AVNNConv2dToLinarBridge
from condensers import MappingCondenser  # Or just Linear() if empty

# === Data Preparation ===
tokenizer = TagValueTokenizer()
tokenizer.exclude_columns([0, 1, 2])
data = open("data.csv").read()
tensor = tokenizer.tokenize(data)  # Shape: [B, F, 2]
dim = tensor.shape[1]

# === Dummy Targets (replace with real labels if you have them)
targets = torch.linspace(0, 1, steps=tensor.size(0)).unsqueeze(1)  # e.g., HP scaled

# === Dataset Split
dataset = TensorDataset(tensor.float(), targets)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# === Model Definition
class AVNNModel(Module):
    def __init__(self, dim, conv_shape=(5, 3, 3)):
        super().__init__()
        c, h, w = conv_shape
        self.model = Sequential(
            AVNNType1Linear(input_dim=dim, output_dim=dim * 2),
            AVNNType2Linear(input_dim=dim * 2, output_dim=dim),
            AVNNLinearBlock(input_dim=dim),
            AVNNLinearToConv2dBridge(c, h, w),
            AVNNConv2dBlock(in_channels=c, out_channels=c),
            AVNNConv2dToLinarBridge(dim=dim, channels=c, height=h, width=w),
            AVNNLinearBlock(input_dim=dim),
            MappingCondenser(),
            Linear(dim * 2, 1)
        )

    def forward(self, x):
        return self.model(x)

# === Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AVNNModel(dim).to(device)
optimizer = Adam(model.parameters(), lr=1e-3)
loss_fn = MSELoss()

# === Training Loop
EPOCHS = 50
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_batch.size(0)

    avg_train_loss = total_loss / train_size

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            pred = model(x_val)
            val_loss += loss_fn(pred, y_val).item() * x_val.size(0)

    avg_val_loss = val_loss / val_size
    print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    with torch.no_grad():
        pred_sample = model(x_val[:5])
        print("Sample preds:", pred_sample.squeeze().cpu().numpy())
        print("Sample targets:", y_val[:5].squeeze().cpu().numpy())

