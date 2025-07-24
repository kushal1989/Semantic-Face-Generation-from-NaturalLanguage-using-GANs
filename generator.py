import torch
import torch.nn as nn
from torchvision.utils import save_image
from io import BytesIO
from PIL import Image

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 3 * 64 * 64),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 3, 64, 64)
        return x

def load_model(model_path='final_generator.pth', device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Generator().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def generate_image(model, embedding, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding = embedding.to(device).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        fake_img = model(embedding)
    fake_img = (fake_img + 1) / 2  # Rescale to [0, 1]

    # Save to BytesIO and return it
    buffer = BytesIO()
    save_image(fake_img, buffer, format='PNG')
    buffer.seek(0)
    return buffer
