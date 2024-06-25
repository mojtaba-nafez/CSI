import numpy as np
import torch
import torch.nn.functional as F
import random
import models.transform_layers as TL
from datasets.cutpast_transformation import *

class NegativePairGenerator:
    def __init__(self, probabilities = {'rotation': 0.0, 'cutperm': 0.0, 'cutout': 0.001, 'cutpaste': 0.999}):
        self.probabilities = probabilities
        
        self.rotation_shift = TL.Rotation()
        self.cutperm_shift = TL.CutPerm()
        self.cutpaste_shift = CutPasteUnion()
        self.aug_to_func = {'rotation': self.apply_rotation, 'cutperm': self.apply_cutperm, 'cutout': self.apply_cutout, 'cutpaste': self.apply_cutpaste}

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
        image[:, mask_x:mask_x + mask_size[0], mask_y:mask_y + mask_size[1]] = 0
        return image
    
    def apply_cutpaste(self, img):
        # input:torch.rand(3, 224, 224)
        # output:torch.rand(3, 224, 224)
        return self.cutpaste_shift(img)

    def create_negative_pair(self, batch_image):   
        augs = list(self.probabilities.keys())
        probs = list(self.probabilities.values())            
        batch_transforms = np.random.choice(augs, size=batch_image.shape[0], p=probs)
        neg_pair = torch.stack([self.aug_to_func[batch_transforms[i]](img) for i, img in enumerate(batch_image)])
        return neg_pair