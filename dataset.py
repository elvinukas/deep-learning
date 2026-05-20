
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image


class LocalColorizationDataset(Dataset):
    def __init__(self, image_dir, limit=None):
        self.image_dir = image_dir

        # Gather all valid files from disk
        all_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_filenames = all_files[:limit] if limit else all_files

        # Image transformations
        self.target_transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])

        self.input_transform = T.Compose([
            T.Resize((256, 256)),
            T.Grayscale(num_output_channels=1),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])

        # Load and verify image mode
        img = Image.open(img_path).convert('RGB')

        target = self.target_transform(img)  # 3x256x256
        gray = self.input_transform(img)  # 1x256x256

        return gray, target