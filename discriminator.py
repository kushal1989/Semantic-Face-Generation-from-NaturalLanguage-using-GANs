import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, img_channels=3, text_dim=512, feature_maps=64):
        super(Discriminator, self).__init__()

        self.image_net = nn.Sequential(
            nn.Conv2d(img_channels, feature_maps, 4, 2, 1),  # 128x128
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1),  # 64x64
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1),  # 32x32
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1),  # 16x16
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps * 8, feature_maps * 8, 4, 2, 1),  # 8x8
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.text_project = nn.Linear(text_dim, feature_maps * 8 * 8 * 8)
        self.out_layer = nn.Sequential(
            nn.Conv2d(feature_maps * 16, 1, kernel_size=4),
            nn.Sigmoid()
        )

    def forward(self, image, text_embedding):
        img_feat = self.image_net(image)  # (B, 512, 8, 8)

        txt_feat = self.text_project(text_embedding).view(-1, 512, 8, 8)
        combined = torch.cat([img_feat, txt_feat], dim=1)  # (B, 1024, 8, 8)

        out = self.out_layer(combined)
        return out.view(-1)
