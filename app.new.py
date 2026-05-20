from __future__ import annotations

import os
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import gradio as gr
from PIL import Image

class ColoringModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.steps1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.steps2 = nn.Sequential(
            nn.MaxPool2d(2, 2), # 256x256 -> 128x128
            nn.Conv2d(16, 32, kernel_size=3, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.steps3 = nn.Sequential(
            nn.MaxPool2d(2, 2), # 128x128 -> 64x64
            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.steps4 = nn.Sequential(
            nn.MaxPool2d(2, 2), # 64x64 -> 32x32
            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.steps5 = nn.Sequential(
            nn.MaxPool2d(2, 2), # 32x32 -> 16x16
            nn.Conv2d(128, 256, kernel_size=3, padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2), # 16x16 -> 32x32
        )

        self.steps6 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2), # 32x32 -> 64x64
        )

        self.steps7 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2), # 64x64 -> 128x128
        )

        self.steps8 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Upsample(scale_factor=2), # 128x128 -> 256x256
        )

        self.steps9 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, padding='same'),
            nn.Sigmoid()
        )

    def forward(self, x):
        block1 = self.steps1(x)
        block2 = self.steps2(block1)
        block3 = self.steps3(block2)
        block4 = self.steps4(block3)
        block5 = self.steps5(block4)

        block6 = self.steps6(torch.cat([block5, block4], dim=1))
        block7 = self.steps7(torch.cat([block6, block3], dim=1))
        block8 = self.steps8(torch.cat([block7, block2], dim=1))
        block9 = self.steps9(torch.cat([block8, block1], dim=1))

        return block9

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ColoringModel().to(device)
MODEL_CKPT = "model_color.pth"

if os.path.exists(MODEL_CKPT):
    model.load_state_dict(torch.load(MODEL_CKPT, map_location=device))
model.eval()

input_transform = T.Compose([
    T.Resize((256, 256)),
    T.Grayscale(num_output_channels=1),
    T.ToTensor()
])

def predict(uploaded_image: Image.Image):
    if uploaded_image is None:
        return None, None

    gray_tensor = input_transform(uploaded_image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(gray_tensor)
    
    pred_img = pred.squeeze(0).cpu().permute(1, 2, 0).numpy()
    pred_img = np.clip(pred_img, 0, 1) 
    pred_img = (pred_img * 255.0).astype(np.uint8)
    
    gray_img = gray_tensor.squeeze(0).squeeze(0).cpu().numpy()
    gray_img = np.clip(gray_img, 0, 1)
    gray_img = (gray_img * 255.0).astype(np.uint8)

    return Image.fromarray(gray_img, mode='L'), Image.fromarray(pred_img)

with gr.Blocks() as demo:
    gr.Markdown("# Image Colorization Predictor")
    gr.Markdown("Upload a grayscale or color image to see how the model colorizes it.")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Input Image")
        with gr.Column():
            gray_image = gr.Image(type="pil", label="Grayscale Image (Model Input)")
            output_image = gr.Image(type="pil", label="Colorized Image")
            run_button = gr.Button("Colorize")

    run_button.click(predict, inputs=[image_input], outputs=[gray_image, output_image])
    image_input.change(predict, inputs=[image_input], outputs=[gray_image, output_image])


if __name__ == "__main__":
    demo.launch()
