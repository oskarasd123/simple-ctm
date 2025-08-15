import torch
from torch import nn, Tensor, optim
import torch.nn.functional as F
from model import ImageExtractor, CTMImageAttention, CTM, SimpleCTM, MoblileImageExtractor
from dataloader import CropedImagePointDataset
import torchvision
from torchvision.transforms import Compose, ToTensor, Lambda, Resize, Normalize
import random
import numpy as np
from some_utils import ExpDecay
from torch.utils.data import DataLoader

steps = 2000
neurons = 512
batch_size = 128
dataset_fraction = 1
lr = 2e-3
save_path = "model_ff.pt"

image_extractor = ImageExtractor().cuda() # 3 channels in, 256 channels out, 16x down sampling
dense_ff = nn.Sequential(
        #nn.AdaptiveAvgPool2d((1, 1)), 
        nn.Flatten(1),
        nn.Linear(256*4*4, neurons), nn.LayerNorm(neurons), nn.ReLU(),
        nn.Linear(neurons, neurons), nn.LayerNorm(neurons), nn.ReLU(),
        nn.Linear(neurons, 1, bias=False),
    ).cuda()

image_extractor.train()
dense_ff.train()

dense_ff[-1].weight.data *= 2

dataset = CropedImagePointDataset("images_labels/", dataset_fraction, Compose([ToTensor(), Resize((64, 64))]), 0.2)



def collate_fn(examples : list):
    images = []
    exists = []
    for image, points in examples:
        images.append(image)
        exists.append(1.0 if points.shape[0] > 0 else 0.0) # 50% of images have points
    return torch.stack(images), torch.tensor(exists).view(-1, 1).cuda()


loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


extractor_optim = optim.AdamW(image_extractor.parameters(), lr, (0.9, 0.99))
linear_optim = optim.AdamW(dense_ff.parameters(), lr, (0.8, 0.99))

loss_decay = ExpDecay(2, 0.2, 3)
prediction_rate = ExpDecay(0.5, 0.2, 2)

loss_fn = nn.BCEWithLogitsLoss()

step = 0
for epoch in range(1000000):
    if step > steps:
        break
    for images, exists in loader:
        if step > steps:
            break
        n_correct_predictions = 0
        image_features = image_extractor(images.to(device="cuda"))
        output : Tensor = dense_ff(image_features)
        loss = loss_fn(output, exists)
        #output = F.sigmoid(output)
        #output = output * 0.999 + 0.0005
        #loss = (-(output).log() * target - (1-output).log() * (1-target)).mean()

        #loss += 0.1/(output.std()+1) # output std still collapses to 0 after some steps


        correct_predictions = ((output > 0) == (exists > 0.5)).float().mean()

        loss.backward()
        extractor_optim.step()
        linear_optim.step()

        extractor_optim.zero_grad()
        linear_optim.zero_grad()

        step += 1
        loss_decay.update(loss.item())
        prediction_rate.update(correct_predictions.item())
        print(f"epoch: {epoch} \tstep: {step} \tloss: {loss_decay.value:.4} \toutput mean: {output.mean().item():.04} \toutput std: {output.std().item():.04} \tcorrect: {prediction_rate.value:.03}")
        if prediction_rate.value > 0.98:
            print("98% prediction rate reached, terminating")
            step = steps+1
            break



torch.save({
    "image_extractor" : image_extractor.state_dict(),
    "dense_ff" : dense_ff.state_dict(),
}, save_path)
print("models saved")

