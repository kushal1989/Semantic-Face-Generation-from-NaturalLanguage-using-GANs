import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from dataclasses import dataclass
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Ensure current directory is in path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from text_encoder import SentenceEncoder


# === Configuration ===
@dataclass
class Config:
    epochs: int = 20
    image_size: int = 64         # updated to match model input
    initial_size: int = 64       # keep FC output at 64x64
    noise_size: int = 100
    batch_size: int = 64
    num_channels: int = 3
    device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cfg = Config()

# === Dataset path ===
IMAGE_FOLDER = r"C:\MAJOR\DATASET\img_align_celeba\img_align_celeba"
CAPTION_CSV = r"C:\MAJOR\DATASET\image_captions.csv"

# === Sentence Encoder
sentence_encoder = SentenceEncoder(cfg.device)

# === Dataset Class
class ImageNTextDataset(Dataset):
    def __init__(self, image_folder, caption_csv_path, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        df = pd.read_csv(caption_csv_path)
        df.columns = df.columns.str.strip()

        if 'image_id' not in df.columns or '0' not in df.columns:
            raise ValueError("CSV file must contain 'image_id' and '0' columns.")

        self.image_files = df['image_id'].tolist()
        self.captions = df['0'].tolist()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        real_img_path = os.path.join(self.image_folder, self.image_files[idx])
        real_img = Image.open(real_img_path).convert("RGB")

        wrong_idx = random.randint(0, len(self.image_files) - 1)
        while wrong_idx == idx:
            wrong_idx = random.randint(0, len(self.image_files) - 1)
        wrong_img_path = os.path.join(self.image_folder, self.image_files[wrong_idx])
        wrong_img = Image.open(wrong_img_path).convert("RGB")

        text = self.captions[idx]
        text_emb = sentence_encoder.encode([text])[0]

        if self.transform:
            real_img = self.transform(real_img)
            wrong_img = self.transform(wrong_img)

        return real_img, text_emb, wrong_img


# === Transform (updated to 64x64)
transform = transforms.Compose([
    transforms.Resize((cfg.image_size, cfg.image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# === Dataloader
dataset = ImageNTextDataset(IMAGE_FOLDER, CAPTION_CSV, transform=transform)
dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

# Detect embedding size
sample_embed = sentence_encoder.encode(["test sentence"])[0]
embedding_size = sample_embed.shape[0]
print(f"Detected text embedding size: {embedding_size}")


# === Models
class Generator(nn.Module):
    def __init__(self, text_embed_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(cfg.noise_size + text_embed_size, 1024),
            nn.ReLU(True),
            nn.Linear(1024, cfg.num_channels * cfg.initial_size * cfg.initial_size),
            nn.Tanh()
        )

    def forward(self, noise, text_embedding):
        x = torch.cat([noise, text_embedding], 1)
        x = self.model(x)
        return x.view(-1, cfg.num_channels, cfg.initial_size, cfg.initial_size)


class Discriminator(nn.Module):
    def __init__(self, text_embed_size):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cfg.num_channels, 128, 4, 2, 1),   # 128 → 64
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),                # 64 → 32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),                # 32 → 16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1024, 4, 2, 1),               # 16 → 8
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1024, 4, 2, 1),              # 8 → 4
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024 * 4 * 4 + text_embed_size, 1024),  # 16384 + text_embed
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, img, text_embedding):
        x = self.conv(img).view(img.size(0), -1)
        x = torch.cat([x, text_embedding], dim=1)
        return self.fc(x)




# === Training Loop
def train():
    generator = Generator(embedding_size).to(cfg.device)
    discriminator = Discriminator(embedding_size).to(cfg.device)

    loss_fn = nn.BCELoss()
    optim_G = optim.Adam(generator.parameters(), lr=0.0002)
    optim_D = optim.Adam(discriminator.parameters(), lr=0.0002)

    best_g_loss = float('inf')
    best_d_loss = float('inf')
    epochs_no_improve = 0
    early_stop_patience = 3

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(cfg.epochs):
        g_loss_total = 0
        d_loss_total = 0

        for i, (real_imgs, text_embs, wrong_imgs) in enumerate(tqdm(dataloader)):
            real_imgs = real_imgs.to(cfg.device)
            text_embs = text_embs.to(cfg.device)

            valid = torch.ones(real_imgs.size(0), 1).to(cfg.device)
            fake = torch.zeros(real_imgs.size(0), 1).to(cfg.device)

            # === Generator ===
            noise = torch.randn(real_imgs.size(0), cfg.noise_size).to(cfg.device)
            gen_imgs = generator(noise, text_embs)
            g_loss = loss_fn(discriminator(gen_imgs, text_embs), valid)

            optim_G.zero_grad()
            g_loss.backward()
            optim_G.step()

            # === Discriminator ===
            real_loss = loss_fn(discriminator(real_imgs, text_embs), valid)
            fake_loss = loss_fn(discriminator(gen_imgs.detach(), text_embs), fake)
            d_loss = (real_loss + fake_loss) / 2

            optim_D.zero_grad()
            d_loss.backward()
            optim_D.step()

            g_loss_total += g_loss.item()
            d_loss_total += d_loss.item()

            if i % 100 == 0:
                print(
                    f"[Epoch {epoch + 1}/{cfg.epochs}] [Batch {i}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]"
                )

        avg_g_loss = g_loss_total / len(dataloader)
        avg_d_loss = d_loss_total / len(dataloader)

        print(f"Epoch {epoch + 1} completed. Avg G Loss: {avg_g_loss:.4f}, Avg D Loss: {avg_d_loss:.4f}")

        # === Checkpointing ===
        if avg_g_loss < best_g_loss:
            best_g_loss = avg_g_loss
            torch.save(generator.state_dict(), "checkpoints/best_generator.pth")
            print("Saved best generator checkpoint.")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if avg_d_loss < best_d_loss:
            best_d_loss = avg_d_loss
            torch.save(discriminator.state_dict(), "checkpoints/best_discriminator.pth")
            print("Saved best discriminator checkpoint.")

        # === Early Stopping ===
        if epochs_no_improve >= early_stop_patience:
            print(f"No improvement in generator loss for {early_stop_patience} epochs. Early stopping.")
            break

    # Save final models
    torch.save(generator.state_dict(), "checkpoints/final_generator.pth")
    torch.save(discriminator.state_dict(), "checkpoints/final_discriminator.pth")
    print("Saved final models.")

if __name__ == "__main__":
    train()
