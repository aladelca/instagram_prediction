import torch
from torch import nn
from torchvision import models


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)  # (batch, 256)


class MultiModalRegressor(nn.Module):
    def __init__(self, text_dim: int, meta_dim: int):
        super().__init__()
        self.image_encoder = SimpleCNN()
        self.image_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.text_head = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.meta_head = nn.Sequential(
            nn.Linear(meta_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Softplus(),  # mantiene salida positiva y gradientes no nulos
        )

    def forward(self, images, text_vec, meta_vec):
        if len(images) == 0:
            raise ValueError("images list is empty")
        image_batch = torch.stack(images)  # (n,3,500,500)
        img_feats = self.image_encoder(image_batch)  # (n,256)
        img_mean = img_feats.mean(dim=0)
        img_out = self.image_head(img_mean)

        text_out = self.text_head(text_vec)
        meta_out = self.meta_head(meta_vec)

        fused = torch.cat([img_out, text_out, meta_out], dim=-1)
        return self.fusion(fused).squeeze(-1)
