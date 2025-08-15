import torch
from torch import nn, Tensor, optim
from dataloader import CropedImagePointDataset, ImagePointDataset
from model import ImageExtractor, MoblileImageExtractor, CTMImageAttention, CTM, SimpleCTM
import torchvision
from torchvision.transforms.v2 import Compose, ToImage, ToDtype, Lambda, Resize, Normalize
from torch.utils.data import DataLoader
from train_utils import mapped_loss, PointMatchingLoss
import random
import numpy as np
import os

steps = 5000
neurons = 512
batch_size = 64
dataset_fraction = 1
history_size = 20
lr = 1e-3
ticks = 40
save_path = "model_512.pt"
load_checkpoint = True


image_extractor = ImageExtractor().cuda() # 3 channels in, 256 channels out, 16x down sampling
image_attention = CTMImageAttention(image_extractor.out_channels, neurons, neurons).cuda()
ctm = CTM(neurons, history_size, neurons, 1).cuda()
output_linear = nn.Linear(neurons, 3 * 20).cuda() # output is 10 points, 10 conficence values


if load_checkpoint and os.path.exists(save_path):
    checkpoint = torch.load(save_path, weights_only=True)
    image_extractor.load_state_dict(checkpoint["image_extractor"])
    image_attention.load_state_dict(checkpoint["image_attention"])
    ctm.load_state_dict(checkpoint["ctm"])
    output_linear.load_state_dict(checkpoint["output_linear"])
else:
    load_checkpoint = False

dataset = CropedImagePointDataset("images_labels/", 1, Compose([ToImage(), ToDtype(torch.float, True), Resize((256, 256))]), 0.2, True)
indicies = list(range(len(dataset)))
random.shuffle(indicies)

def collate_fn(examples : list):
    images = []
    true_points = []
    max_points = 0
    for image, points in examples:
        images.append(image)
        max_points = max(max_points, points.shape[0])
    for image, points in examples:
        true_points.append(list(points) + [(float("nan"), float("nan")) for i in range(max_points - points.shape[0])])
    return torch.stack(images), torch.tensor(np.array(true_points)).cuda()


data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)



loss_fn = PointMatchingLoss(1, 0.1, 0.02)

extractor_optim = optim.AdamW(image_extractor.parameters(), 1e-3)
attention_optim = optim.AdamW(image_attention.parameters(), 1e-3, (0.8, 0.95))
ctm_optim = optim.AdamW(ctm.parameters(), 1e-3)
output_optim = optim.Adam(output_linear.parameters(), 1e-3, weight_decay=0.0)

if load_checkpoint and "optimizers" in checkpoint:
    optim_checkpoint = checkpoint["optimizers"]
    extractor_optim.load_state_dict(optim_checkpoint["extractor_optim"])
    attention_optim.load_state_dict(optim_checkpoint["attention_optim"])
    ctm_optim.load_state_dict(optim_checkpoint["ctm_optim"])
    output_optim.load_state_dict(optim_checkpoint["output_optim"])

step = 0
try:
    for epoch in range(1000000):
        if step > steps:
            break
        for i, (images, points) in enumerate(data_loader):
            if step > steps:
                break
            if points.shape[0] == 0:
                continue
            batches = images.size(0)
            images = images.cuda()
            points = points.cuda()*2-1 # from 0 - 1 to -1 - 1

            image_features = image_extractor(images)
            latent = ctm.reset(batches)
            loss = torch.zeros((1,), device="cuda")
            n_correct_predictions = 0
            for tick in range(ticks):
                ctm_input = image_attention(image_features, latent)
                latent = ctm(ctm_input)
                output : Tensor = output_linear(latent).reshape(batches, -1, 3)
                tick_loss, correct_predictions = loss_fn(output, points)
                n_correct_predictions += correct_predictions
                loss += tick_loss * (tick/ticks)
                
            
            loss = loss/ticks


            loss.backward()
            extractor_optim.step()
            attention_optim.step()
            ctm_optim.step()
            output_optim.step()

            extractor_optim.zero_grad()
            attention_optim.zero_grad()
            ctm_optim.zero_grad()
            output_optim.zero_grad()
            step += 1
            nr_points = (1-points[:,:,0].isnan().float()).sum()
            print(f"epoch: {epoch} \tstep: {step} \tloss: {loss.item():.4} \tnr points: {nr_points/batch_size} \tprediction rate: {n_correct_predictions/ticks/nr_points:.3}")
except KeyboardInterrupt:
    yes_or_no = input("save model(Y/n): ").lower() != "n"
    if yes_or_no:
        torch.save({
            "ctm" : ctm.state_dict(),
            "image_extractor" : image_extractor.state_dict(),
            "image_attention" : image_attention.state_dict(),
            "output_linear" : output_linear.state_dict(),
            "optimizers": {
                "extractor_optim": extractor_optim.state_dict(),
                "attention_optim": attention_optim.state_dict(),
                "ctm_optim": ctm_optim.state_dict(),
                "output_optim": output_optim.state_dict(),
            }
        }, save_path)
        print("models saved")
    exit()

torch.save({
    "ctm" : ctm.state_dict(),
    "image_extractor" : image_extractor.state_dict(),
    "image_attention" : image_attention.state_dict(),
    "output_linear" : output_linear.state_dict(),
    "optimizers": {
        "extractor_optim": extractor_optim.state_dict(),
        "attention_optim": attention_optim.state_dict(),
        "ctm_optim": ctm_optim.state_dict(),
        "output_optim": output_optim.state_dict(),
    }
}, save_path)
print("models saved")

