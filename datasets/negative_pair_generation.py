import numpy as np
import torch
import torch.nn.functional as F
import random
import models.transform_layers as TL
import json
from torchvision import datasets, transforms
from glob import glob
from PIL import Image


class ImageNetExposure(Dataset):
    def __init__(self, root, count, transform=None):
        self.transform = transform
        image_files = glob(os.path.join(root, 'train', "*", "images", "*.JPEG"))
        if count==-1:
            final_length = len(image_files)
        else:
            random.shuffle(image_files)
            final_length = min(len(image_files), count)
        self.image_files = image_files[:final_length]
        self.image_files.sort(key=lambda y: y.lower())

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, -1

    def __len__(self):
        return len(self.image_files)
        
class NegativePairGenerator:
    def __init__(self):
        tiny_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.AutoAugment(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
        ])
        imagenet_exposure = ImageNetExposure(root='./tiny-imagenet-200', count=10000, transform=tiny_transform)
        self.imagenet_exposure_loader = DataLoader(imagenet_exposure, shuffle=True, batch_size=1024)

        # self.probabilities = {'rotation': 0.0, 'cutperm': 0.0, 'cutout': 0.01, 'cutpaste': 0.99}
        with open('./config.json', 'r') as json_file:
            self.probabilities = json.load(json_file)
        
        self.rotation_shift = TL.Rotation()
        self.cutperm_shift = TL.CutPerm()
        self.cutpaste_shift = TL.CutPasteLayer()
        self.aug_to_func = {'rotation': self.apply_rotation, 'cutperm': self.apply_cutperm, 'cutout': self.apply_cutout, 'cutpaste': self.apply_cutpaste}
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    def apply_rotation(self, img):
        # input:torch.rand(3, 224, 224)
        # output:torch.rand(3, 224, 224)
        return self.rotation_shift(img.unsqueeze(0), np.random.randint(1, 4)).squeeze()
    
    def apply_cutperm(self, img):
        # input:torch.rand(3, 224, 224)
        # output:torch.rand(3, 224, 224)
        return self.cutperm_shift(img.unsqueeze(0), np.random.randint(1, 4)).squeeze()

    def apply_cutout(self, image):
        # input:torch.rand(3, 224, 224)
        # output:torch.rand(3, 224, 224)
        _, h, w = image.shape
        mask_size = (np.random.randint(h // 6, h // 4), np.random.randint(w // 6, w // 4))
        mask_x = random.randint(0, h - mask_size[0])
        mask_y = random.randint(0, w - mask_size[1])
        image[:, mask_x:mask_x + mask_size[0], mask_y:mask_y + mask_size[1]] = 0.0
        return image
    
    def apply_cutpaste(self, img):
        # input:torch.rand(3, 224, 224)
        # output:torch.rand(3, 224, 224)
        return self.cutpaste_shift(img.unsqueeze(0)).squeeze()

    def create_negative_pair(self, batch_image):
        x, _ = next(iter(self.imagenet_exposure_loader))
        return x.to(self.device)
        
        batch_image = batch_image.to(self.device)   
        augs = list(self.probabilities.keys())
        probs = list(self.probabilities.values())            
        batch_transforms = np.random.choice(augs, size=batch_image.shape[0], p=probs)
        neg_pair = torch.stack([self.aug_to_func[batch_transforms[i]](img) for i, img in enumerate(batch_image)])
        return neg_pair