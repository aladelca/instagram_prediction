from pathlib import Path
from typing import List, Dict

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class InstagramPostDataset(Dataset):
    def __init__(self, processed_path: str, device=None):
        # torch 2.6 switched default to weights_only=True; here we need full objects
        self.data: List[Dict] = torch.load(processed_path, weights_only=False)
        self.device = device or torch.device("cpu")
        self.transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        images = []
        for img_path in sample["image_paths"]:
            with Image.open(img_path).convert("RGB") as img:
                images.append(self.transform(img))
        text_vec = torch.tensor(sample["text_vector"], dtype=torch.float32)
        meta_vec = torch.tensor(sample["meta_vector"], dtype=torch.float32)
        target = torch.tensor(sample["target"], dtype=torch.float32)
        return {
            "images": images,
            "text": text_vec,
            "meta": meta_vec,
            "target": target,
            "id": sample["id"],
        }


def simple_collate(batch):
    return batch
