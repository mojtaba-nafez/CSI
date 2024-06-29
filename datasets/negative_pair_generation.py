import numpy as np
import torch
import torch.nn.functional as F
import random
import models.transform_layers as TL
import json
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

from datasets.custom_datasets import ImageNetMixUp

class NegativePairGenerator:
    def __init__(self, P):
        # self.probabilities = {'rotation': 0.0, 'cutperm': 0.0, 'cutout': 0.01, 'cutpaste': 0.99}
        with open('./config.json', 'r') as json_file:
            self.probabilities = json.load(json_file)
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
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
        self.rotation_shift = self.rotation_shift.to(self.device)
        self.cutperm_shift = self.cutperm_shift.to(self.device)
        self.cutpaste_shift = self.cutpaste_shift.to(self.device)
        
        trans = transforms.Compose([
                transforms.Resize((P.image_size, P.image_size)),
                transforms.AutoAugment(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
        ])
        self.mixup_dataset = ImageNetMixUp(root='./tiny-imagenet-200', count=P.normal_data_count, transform=trans)
        self.mixup_loader = DataLoader(self.mixup_dataset, shuffle=True, batch_size=1)
        self.mixup_iter = iter(self.mixup_loader)


        self.aug_to_func = {'mixup': self.apply_mixup, 'elastic': self.apply_elastic, 'rotation': self.apply_rotation, 'cutperm': self.apply_cutperm, 'cutout': self.apply_cutout, 'cutpaste': self.apply_cutpaste}

    def apply_mixup(self, img):
        print(img.shape)
        print(self.mixup_dataset[0].shape)
        print(len(self.mixup_dataset))
        try:
            mixed_img, _ = next(self.mixup_iter)
        except:
            self.mixup_iter = iter(self.mixup_loader)
            mixed_img, _ = next(self.mixup_iter)
        mixed_img = mixed_img.to(self.device)
        lam = torch.tensor(random.uniform(0.4, 0.8)).to(self.device)
        return lam * img + (1 - lam) * mixed_img

    def apply_rotation(self, img):
        # input:torch.rand(3, 224, 224)
        # output:torch.rand(3, 224, 224)
        # img = self.auto_aug(img)
        # img = self.elastic_aug(img)
        img = self.rotation_shift(img.unsqueeze(0), np.random.randint(1, 4))
        # img = self.rotation_shift(img.unsqueeze(0), 2)
        return img.squeeze().to(self.device)

    def apply_elastic(self, img):
        # input:torch.rand(3, 224, 224)
        # output:torch.rand(3, 224, 224)
        #img = self.auto_aug(img)
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