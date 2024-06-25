from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from common.common import parse_args
import models.classifier as C
from datasets import set_dataset_count, get_dataset, get_superclass_list, get_subclass_dataset
from utils.utils import load_checkpoint, get_loader_unique_label, count_parameters

P = parse_args()
cls_list = get_superclass_list(P.dataset)
anomaly_labels = [elem for elem in cls_list if elem not in [P.normal_label]]

if torch.cuda.is_available():
    torch.cuda.set_device(P.local_rank)
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
P.multi_gpu = False

image_size_ = (P.image_size, P.image_size, 3)
train_set, test_set, image_size, n_classes = get_dataset(P, dataset=P.dataset, download=True, image_size=image_size_, labels=[P.normal_label])
P.image_size = image_size
P.n_classes = n_classes
print("full test set:", len(test_set))
print("full train set:", len(train_set))
full_test_set = deepcopy(test_set)  # test set of full classes

if P.dataset=='ISIC2018' or P.dataset=='mvtecad' or P.dataset=='cifar10-versus-100' or P.dataset=='cifar100-versus-10':
    train_set = set_dataset_count(train_set, count=P.main_count)
    test_set = get_subclass_dataset(P, test_set, classes=[0])
else:
    train_set = get_subclass_dataset(P, train_set, classes=[P.normal_label], count=P.main_count)
    test_set = get_subclass_dataset(P, test_set, classes=[P.normal_label])
        
print("number of normal test set:", len(test_set))
print("number of normal train set:", len(train_set))

kwargs = {'pin_memory': False, 'num_workers': 4}

train_loader = DataLoader(train_set, shuffle=True, batch_size=P.batch_size, **kwargs)
test_loader = DataLoader(test_set, shuffle=False, batch_size=P.test_batch_size, **kwargs)

print("len train_set", len(train_set))
print("len test_set", len(test_set))

print("Unique labels(test_loader):", get_loader_unique_label(test_loader))
print("Unique labels(train_loader):", get_loader_unique_label(train_loader))

if P.dataset=='ISIC2018' or P.dataset=="mvtecad" or P.dataset=='cifar10-versus-100' or P.dataset=='cifar100-versus-10':
    anomaly_labels = [1]

ood_test_loader = dict()
for ood in anomaly_labels:
    ood_test_set = get_subclass_dataset(P, full_test_set, classes=ood)
    ood = f'one_class_{ood}'
    print(f"testset anomaly(class {ood}):", len(ood_test_set))
    ood_test_loader[ood] = DataLoader(ood_test_set, shuffle=False, batch_size=P.test_batch_size, **kwargs)
    print("Unique labels(ood_test_loader):", get_loader_unique_label(ood_test_loader[ood]))


simclr_aug = C.get_simclr_augmentation(P, image_size=P.image_size).to(device)

model = C.get_classifier(P.model, n_classes=P.n_classes, activation=P.activation_function, mean=P.noise_mean, std=P.noise_std, noise_scale=P.noise_scale, noist_probability=P.noist_probability, freezing_layer=P.freezing_layer).to(device)
model = C.get_shift_classifer(model, 2).to(device)

criterion = nn.CrossEntropyLoss().to(device)

if P.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=P.lr_init, momentum=0.9, weight_decay=P.weight_decay)
    lr_decay_gamma = 0.1
elif P.optimizer == 'lars':
    from torchlars import LARS
    base_optimizer = optim.SGD(model.parameters(), lr=P.lr_init, momentum=0.9, weight_decay=P.weight_decay)
    optimizer = LARS(base_optimizer, eps=1e-8, trust_coef=0.001)
    lr_decay_gamma = 0.1
else:
    raise NotImplementedError()

if P.lr_scheduler == 'cosine':
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, P.epochs)
elif P.lr_scheduler == 'step_decay':
    milestones = [int(0.5 * P.epochs), int(0.75 * P.epochs)]
    scheduler = lr_scheduler.MultiStepLR(optimizer, gamma=lr_decay_gamma, milestones=milestones)
else:
    raise NotImplementedError()

from training.scheduler import GradualWarmupScheduler
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=10.0, total_epoch=P.warmup, after_scheduler=scheduler)

if P.resume_path is not None:
    resume = True
    model_state, optim_state, config = load_checkpoint(P.resume_path, mode='last')
    model.load_state_dict(model_state, strict=not P.no_strict)
    optimizer.load_state_dict(optim_state)
    start_epoch = config['epoch']
    error = 100.0
else:
    resume = False
    start_epoch = 1
    error = 100.0

count_parameters(model)