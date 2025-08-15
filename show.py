import torch
from torch import nn, Tensor, optim
from dataloader import CropedImagePointDataset, ImagePointDataset
from model import ImageExtractor, CTMImageAttention, CTM, SimpleCTM
import torchvision
from torchvision.transforms import Compose, ToTensor, Lambda, Resize
from torch.utils.data import DataLoader
import cv2
import numpy as np
import random

neurons = 512
ticks = 200
history_size = 20
model_path = "model_512.pt"

image_extractor = ImageExtractor().cuda() # 3 channels in, 256 channels out, 16x down sampling
image_attention = CTMImageAttention(image_extractor.out_channels, neurons, neurons).cuda()
ctm = CTM(neurons, history_size, neurons, 1).cuda()
output_linear = nn.Linear(neurons, 3 * 20).cuda() # output is 10 points, 10 conficence values


dataset = CropedImagePointDataset("images_labels/", 1, Compose([ToTensor(), Resize((256, 256))]), 0.2, True)

model_dict = torch.load(model_path, weights_only = True)

ctm.load_state_dict(model_dict["ctm"])
image_extractor.load_state_dict(model_dict["image_extractor"])
image_attention.load_state_dict(model_dict["image_attention"])
output_linear.load_state_dict(model_dict["output_linear"])

show_attention = False
_running = True
while _running:
    image, true_points = random.choice(dataset)

    show_image = image.squeeze(0).permute((1, 2, 0)).numpy(force = True).copy()
    show_image = (show_image*255).clip(0, 255).astype(np.uint8)
    show_image = cv2.resize(show_image, (1024, 1024))

    image_features = image_extractor(image.unsqueeze(0).to(device="cuda"))
    latents = [ctm.reset(1)]
    loss = torch.zeros((1,), device="cuda")
    tick = 0
    calculated_tick = 0
    while True:
        print(f"\rtick: {tick}  ", end = "")
        image = show_image.copy()
        ctm_input, attention_map_torch = image_attention(image_features, latents[tick], True)
        if calculated_tick <= tick:
            latents.append(ctm(ctm_input))
            calculated_tick += 1
        output : Tensor = output_linear(latents[tick+1])
        output = output.reshape(-1, 3).numpy(force = True)
        points = output[:,:2]
        confidences = output[:,2]
        normalized_confidences = confidences - confidences.mean()
        normalized_confidences = normalized_confidences / normalized_confidences.std()

        attention_map = attention_map_torch.permute((1, 2, 0)).sqrt().numpy(force=True).repeat(3, -1)
        attention_map = cv2.resize(attention_map, (1024, 1024))**0.5

        points = (points+1)/2 * 1024

        if show_attention:
            image = (image * attention_map).clip(0, 255).astype(np.uint8)
        for point, normalized_confidence, confidence in zip(points, normalized_confidences, confidences):
            radius = max(((normalized_confidence + 3) * 4).astype(int), 1)
            
            point = point.astype(int)
            image = cv2.circle(image, (point[0], point[1]), radius, (255, 255, 0), 2 if confidence < 2 else 4)
            image = cv2.circle(image, (point[0], point[1]), radius + 1, (0, 128, 255), 2)


        cv2.imshow("test", image)
        key = cv2.waitKey(-1) & 0xff
        if key == 27:
            _running = False
            break
        elif key == ord(" "):
            break
        elif key == ord("r"):
            show_attention = not show_attention
        elif key == ord("q"):
            if tick > 1:
                tick -= 1
        else:
            tick += 1




