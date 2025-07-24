import torch
import torch.nn as nn
import streamlit as st
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from PIL import Image
import io
class CFG:
    noise_size = 100
    text_embed_size = 256  # or 356 if your trained model uses that
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = CFG()
# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEXT_DIM = 256  # Must match training
IMG_SIZE = 64

# Path to the saved generator checkpoint
MODEL_PATH = r"C:\Users\USER\PycharmProjects\MAJOR PROJECT\MODELS\gan_checkpoint.pth"

# Generator class that matches the training setup
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim=100, text_dim=256, img_channels=3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(noise_dim + text_dim, 128 * 8 * 8),  # 356 → 8192
            nn.BatchNorm1d(128 * 8 * 8),
            nn.ReLU(True)
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 8x8 → 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 16x16 → 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, img_channels, 4, 2, 1),  # 32x32 → 64x64
            nn.Tanh()
        )

    def forward(self, noise, text_embedding):
        x = torch.cat([noise, text_embedding], dim=1)  # [B, 356]
        x = self.fc(x)                                 # [B, 8192]
        x = x.view(-1, 128, 8, 8)                      # [B, 128, 8, 8]
        return self.deconv(x)                          # [B, 3, 64, 64]


# Load generator model from checkpoint
@st.cache_resource
def load_generator():
    model = Generator(text_dim=TEXT_DIM).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['generator'])  # Load only the generator part
    model.eval()
    return model

# Function to generate image
def generate_image(text_embedding, model):
    model.eval()
    with torch.no_grad():
        # Create noise
        noise = torch.randn(1, cfg.noise_size, device=cfg.device)

        # Ensure text_embedding is tensor and on correct device
        if not isinstance(text_embedding, torch.Tensor):
            text_embedding = torch.tensor(text_embedding, dtype=torch.float32)
        text_embedding = text_embedding.to(cfg.device).unsqueeze(0)

        # Generate image
        generated_img = model(noise, text_embedding).cpu().squeeze(0)
        generated_img = (generated_img + 1) / 2  # Denormalize to [0, 1]
    return generated_img


# Streamlit UI
st.title("Semantic Face Generation from Natural Language using GANs")

text_input = st.text_input("Enter a sentence to generate an image:")

if st.button("Generate"):
    if not text_input.strip():
        st.warning("Please enter a sentence.")
    else:
        # Fake placeholder for sentence embedding (replace with your encoder)
        torch.manual_seed(abs(hash(text_input)) % (2**32))
        fake_embedding = torch.randn(TEXT_DIM)

        model = load_generator()
        output_img = generate_image(fake_embedding, model)
        from torchvision.transforms import ToPILImage

        # Convert the output tensor to PIL
        to_pil = ToPILImage()
        output_img = output_img.detach().cpu().clamp(-1, 1)  # If output is [-1, 1] range
        output_img = (output_img + 1) / 2  # Normalize to [0, 1] range
        output_img_pil = to_pil(output_img.squeeze(0))  # Remove batch dimension

        # Display the image
        st.image(output_img_pil, caption="Generated Image", use_column_width=True)
