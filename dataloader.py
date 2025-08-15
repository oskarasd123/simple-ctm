import os
import random
import torch
from torch import Tensor
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2

def load_image(f):
    im = Image.open(f)  
    im.draft('RGB',(1920,1080))
    return np.asarray(im) 

def clamp(x, minimum, maximum):
    return max(min(x, maximum), minimum)

class ImagePointDataset(Dataset):
    def __init__(self, folder_path, density=1.0, transform=None):
        self.folder_path = folder_path
        self.density = density
        self.transform = transform or transforms.ToTensor()
        random.seed(2) # set seed for consistency
        # Collect pairs of image and TSV files
        self.samples = []
        for fname in os.listdir(folder_path):
            if fname.lower().endswith((".jpg", ".jpeg")):
                base = os.path.splitext(fname)[0]
                img_path = os.path.join(folder_path, fname)
                tsv_path = os.path.join(folder_path, base + ".tsv")
                if os.path.exists(tsv_path):
                    if random.random() < density:
                        self.samples.append((img_path, tsv_path))
        random.seed() # unset seed
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor, int]:
        img_path, tsv_path = self.samples[idx]

        # Load image
        #image = np.asarray(Image.open(img_path))
        #image = cv2.imread(img_path)
        image = load_image(img_path)

        points = []
        with open(tsv_path, "r") as file:
            for line in file.readlines():
                x, y = line.split("\t")
                points.append((float(x),float(y)))

        image = self.transform(image)

        return image, np.array(points)

class CropedImagePointDataset(Dataset):
    def __init__(self, folder_path, density=1.0, transform=None, crop_scale = 0.2, only_with_points = False):
        self.folder_path = folder_path
        self.density = density
        self.transform = transform or transforms.ToTensor()
        self.crop_scale = crop_scale
        self.only_with_points = only_with_points
        random.seed(2) # set seed for consistency
        # Collect pairs of image and TSV files
        self.samples = []
        for fname in os.listdir(folder_path):
            if fname.lower().endswith((".jpg", ".jpeg")):
                base = os.path.splitext(fname)[0]
                img_path = os.path.join(folder_path, fname)
                tsv_path = os.path.join(folder_path, base + ".tsv")
                if os.path.exists(tsv_path):
                    if random.random() < density:
                        self.samples.append((img_path, tsv_path))
        random.seed() # unset seed
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor, int]:
        img_path, tsv_path = self.samples[idx]

        # Load image
        #image = np.asarray(Image.open(img_path))
        #image = cv2.imread(img_path)
        image = load_image(img_path)

        points = []
        with open(tsv_path, "r") as file:
            for line in file.readlines():
                x, y = line.split("\t")
                points.append((float(x),float(y)))
        
        
        height, width = image.shape[:2]
        w = h = int(height * self.crop_scale * (1+random.random())) # width, height are the same for a square
        if (self.only_with_points or random.random() < 0.5) and len(points) > 0: # 50% chance to have at least 1 point
            x_p, y_p = random.choice(points)
            x = x_p * width + (random.random()-0.5)*w * 0.9
            y = y_p * height + (random.random()-0.5)*h * 0.9
            x, y = clamp(x, w//2, width-w//2), clamp(y, h//2, height - h//2)
            x1, x2 = x - w/2, x + w/2
            y1, y2 = y - h/2, y + h/2
            x1 /= width # from 0 - width to 0 - 1
            x2 /= width
            y1 /= height
            y2 /= height
            cropped_points = [((p[0]-x1)/(x2-x1), (p[1]-y1)/(y2-y1)) for p in points if x1 < p[0] < x2 and y1 < p[1] < y2]
        else:
            for try_ in range(20):
                x, y = w/2 + random.random() * (width-w), w/2 + random.random() * (height-h) # center position
                x1, x2 = x - w/2, x + w/2
                y1, y2 = y - h/2, y + h/2
                x1 /= width # from 0 - width to 0 - 1
                x2 /= width
                y1 /= height
                y2 /= height
                cropped_points = [((p[0]-x1)/(x2-x1), (p[1]-y1)/(y2-y1)) for p in points if x1 < p[0] < x2 and y1 < p[1] < y2]
                if len(cropped_points) == 0:
                    break

        x1 *= width # from 0 - width to 0 - 1
        x2 *= width
        y1 *= height
        y2 *= height
        x1, x2, y1, y2 = list(map(int, [x1, x2, y1, y2]))
        cropped_image = image[y1:y2, x1:x2, :]
        cropped_image = self.transform(cropped_image)
        cropped_points.sort(key=lambda x : x[0])
        
        return cropped_image, np.array(cropped_points)


if __name__ == "__main__":
    dataset = CropedImagePointDataset("images_labels/", 1, lambda x : x)
    _running = True
    i = 0
    ocupied = 0
    unocupied = 0
    import time
    start_time = time.time()
    for i in range(len(dataset)):
        st = time.time()
        dataset[i]
        tt = time.time() - st
        print(f"index: {i} \tname: {dataset.samples[i]} \ttime: {tt*1000:.04}ms \taverage time: {(time.time()-start_time)/(i+1)*1000:.04}ms")
        pass
    total_time = time.time()-start_time
    print(f"loading takes {total_time/len(dataset)*1000:.04}ms/example")
    while _running:
        image, points = dataset[i]
        for x, y in points:
            x *= image.shape[1]
            y *= image.shape[0]
            x, y = int(x), int(y)
            image = cv2.circle(image, (x, y), 10, (0, 0, 255), -1)
        
        if points.shape[0] == 0:
            unocupied += 1
        else:
            ocupied += 1
        print(f"nr points: {points.shape[0]}, point fraction: {ocupied/(ocupied+unocupied)}%")
        image = cv2.resize(image, (800, 800))
        cv2.imshow("dataset image", image)
        key = cv2.waitKey(-1) & 0xff
        if key == 27 or key == ord("q"): # exit when pressing esc or q
            break

        i += 1
