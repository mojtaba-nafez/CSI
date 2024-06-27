import numpy as np
import torch
import torch.nn.functional as F
import random
import models.transform_layers as TL
import json
from torchvision import transforms

class NegativePairGenerator:
    def __init__(self):
        # self.probabilities = {'rotation': 0.0, 'cutperm': 0.0, 'cutout': 0.01, 'cutpaste': 0.99}
        with open('./config.json', 'r') as json_file:
            self.probabilities = json.load(json_file)

        self.auto_aug = transforms.Compose([
                transforms.ToPILImage(),
                transforms.AutoAugment(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
        self.elastic_aug = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ElasticTransform(alpha=200.0, sigma=7.0),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])

        self.rotation_shift = TL.Rotation()
        self.cutperm_shift = TL.CutPerm()
        self.cutpaste_shift = TL.CutPasteLayer()
        self.aug_to_func = {'elastic': self.apply_elastic, 'rotation': self.apply_rotation, 'cutperm': self.apply_cutperm, 'cutout': self.apply_cutout, 'cutpaste': self.apply_cutpaste}
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    def apply_rotation(self, img):
        # input:torch.rand(3, 224, 224)
        # output:torch.rand(3, 224, 224)
        # img = self.auto_aug(img)
        # img = self.elastic_aug(img)
        # img = self.rotation_shift(img.unsqueeze(0), np.random.randint(1, 4))
        img = self.rotation_shift(img.unsqueeze(0), 2)
        return img.squeeze().to(self.device)

    def apply_elastic(self, img):
        # input:torch.rand(3, 224, 224)
        # output:torch.rand(3, 224, 224)
        return self.elastic_aug(img).to(self.device)
        
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
        batch_image = batch_image.to(self.device)   
        augs = list(self.probabilities.keys())
        probs = list(self.probabilities.values())            
        batch_transforms = np.random.choice(augs, size=batch_image.shape[0], p=probs)
        neg_pair = torch.stack([self.aug_to_func[batch_transforms[i]](img) for i, img in enumerate(batch_image)])
        return neg_pair