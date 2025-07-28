import os
import json
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# AVNN imports (assumes `avnn` package is in PYTHONPATH or installed)
from tokenizers import TagValueTokenizer
from blocks import AVNNConv2dBlock, AVNNConv2dResBlock, AVNNLinearBlock, AVNNResBlock
from condensers import MappingCondenser, FusingCondenser, SeparatingCondenser
from derives import derived_adjustedmean

# ---------------------------------
# Configurations
# ---------------------------------
DATA_FILE = "data.csv"
IMAGE_ROOT = "../pokemon/main-sprites/black-white"
BATCH_SIZE = 16
IMG_SIZE = 96
EPOCHS = 100
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ---------------------------------
# Dataset Definition
# ---------------------------------
class PokemonSpriteDataset(Dataset):
    def __init__(self, annotation_file, image_root, tokenizer, transform=None):
        self.records = [r.split("|") for r in open(annotation_file).read().splitlines()]
        self.root = image_root
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        name, front_p, back_p, *tags = self.records[idx]
        front = Image.open(os.path.join(self.root, front_p)).convert("RGB")
        back  = Image.open(os.path.join(self.root,  back_p)).convert("RGB")
        if self.transform:
            front = self.transform(front)
            back  = self.transform(back)
        # Tokenize tags into AVNN tensor shape [dim, 2]
        avnn_tokens = self.tokenizer.tokenize("|".join(tags))  # returns LongTensor [rows,2]
        return {"front": front, "back": back, "tags": avnn_tokens}

# ---------------------------------
# Prepare transforms, tokenizer, dataset
# ---------------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

tokenizer = TagValueTokenizer(row_delimiter="|", column_separator="|", tag_value_separator=":")
tokenizer.exclude_columns([0])  # no excluded columns

dataset = PokemonSpriteDataset(DATA_FILE, IMAGE_ROOT, tokenizer, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# ---------------------------------
# AVNN-Based UNet-like architecture
# ---------------------------------
class AVNNUNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, tag_dim=None):
        super().__init__()
        # Encoder blocks
        self.enc1 = AVNNConv2dBlock(in_channels, hidden_channels=base_channels)
        self.enc2 = AVNNConv2dResBlock(base_channels, hidden_channels=base_channels*2)
        # Bottleneck
        self.bottleneck = AVNNConv2dBlock(base_channels*2, hidden_channels=base_channels*4)
        # Decoder blocks
        self.dec2 = AVNNConv2dResBlock(base_channels*4, hidden_channels=base_channels*2)
        self.dec1 = AVNNConv2dBlock(base_channels*2, hidden_channels=base_channels)
        # Final projection to 3 channels
        self.final_conv = nn.Conv2d(base_channels, in_channels*2, kernel_size=1)
        # Condensers for bridging tag embeddings
        self.tag_map = MappingCondenser()
        self.tag_fuse = FusingCondenser(input_dim=tag_dim or 163)

    def forward(self, x, tags):
        # x: [B, 3, H, W]
        # tags: [B, dim, 2]
        # Fuse tags into a conditioning vector
        cond = self.tag_fuse(tags)  # [B, dim]
        # (Optionally broadcast cond and concatenate to x)
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        # Bottleneck
        x3 = self.bottleneck(x2)
        # Decoder
        x4 = self.dec2(x3 + x2)
        x5 = self.dec1(x4 + x1)
        # Project back to AVNN tensor
        out = self.final_conv(x5)
        # Reshape to AVNN format: split last dim into [value, meaning]
        B, C2, H, W = out.shape
        out = out.view(B, C2//2, H, W, 2)
        return out

# ---------------------------------
# Training Utilities
# ---------------------------------
def save_checkpoint(model, optimizer, epoch, loss, best_loss):
    state = {"epoch": epoch, "model_state": model.state_dict(),
             "optim_state": optimizer.state_dict(), "loss": loss}
    tmp = os.path.join(CHECKPOINT_DIR, f"tmp-{epoch}.pt")
    torch.save(state, tmp)
    current = os.path.join(CHECKPOINT_DIR, f"avnn-{epoch}-current.pt")
    os.replace(tmp, current)
    if loss < best_loss:
        best_path = os.path.join(CHECKPOINT_DIR, f"avnn-{epoch}-best.pt")
        torch.save(state, best_path)
        best_loss = loss
    # Update metadata
    meta = {"best_loss": best_loss, "best_epoch": epoch}
    with open(os.path.join(CHECKPOINT_DIR, "meta.json"), "w") as fd:
        json.dump(meta, fd)
    return best_loss

# ---------------------------------
# Main Training Loop
# ---------------------------------

def train():
    # Initialize model, optimizer, loss
    model = AVNNUNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    mse = nn.MSELoss()

    # Resume if checkpoint exists
    meta_path = os.path.join(CHECKPOINT_DIR, "meta.json")
    start_epoch, best_loss = 0, float('inf')
    if os.path.isfile(meta_path):
        meta = json.load(open(meta_path))
        best_loss = meta.get("best_loss", float('inf'))
        # find latest current checkpoint
        currents = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith("-current.pt")]
        if currents:
            latest = sorted(currents)[-1]
            state = torch.load(os.path.join(CHECKPOINT_DIR, latest), map_location=DEVICE)
            model.load_state_dict(state["model_state"])
            optimizer.load_state_dict(state["optim_state"])
            start_epoch = state["epoch"] + 1
    
    for epoch in range(start_epoch, EPOCHS):
        running_loss = 0.0
        for batch in dataloader:
            imgs_f = batch["front"].to(DEVICE)
            imgs_b = batch["back"].to(DEVICE)
            tags   = batch["tags"].to(DEVICE)
            # Combine front & back as separate samples or process both
            optimizer.zero_grad()
            # TODO: add forward diffusion noise here
            out_f = model(imgs_f, tags)
            out_b = model(imgs_b, tags)
            # Convert AVNN outputs back to images: pick activator channel
            pred_f = out_f[..., 0]
            pred_b = out_b[..., 0]
            loss = mse(pred_f, imgs_f) + mse(pred_b, imgs_b)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(dataloader)
        best_loss = save_checkpoint(model, optimizer, epoch, epoch_loss, best_loss)
        print(f"Epoch {epoch} | Loss: {epoch_loss:.4f} | Best: {best_loss:.4f}")

if __name__ == "__main__":
    train()
