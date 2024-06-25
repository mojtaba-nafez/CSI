import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from copy import deepcopy

from evals.ood_pre import eval_ood_detection
from args import parse_args
import models.classifier as C
from datasets import (
    set_dataset_count, 
    get_dataset, 
    get_superclass_list, 
    get_subclass_dataset
)
from utils.utils import get_loader_unique_label

def setup_device(local_rank):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return device

def initialize_datasets(P):
    image_size_ = (P.image_size, P.image_size, 3)
    train_set, test_set, image_size, n_classes = get_dataset(
        P, dataset=P.dataset, eval=True, download=True, image_size=image_size_, labels=[P.normal_label]
    )
    P.image_size = image_size
    P.n_classes = n_classes
    full_test_set = deepcopy(test_set)
    
    if P.dataset in {
        'cifar10-versus-other-eval', 
        'cifar100-versus-other-eval', 
        'ISIC2018', 
        'mvtecad', 
        'cifar10-versus-100', 
        'cifar100-versus-10'
    }:
        train_set = set_dataset_count(train_set, count=P.main_count)
        test_set = get_subclass_dataset(P, test_set, classes=[0])
    else:
        train_set = get_subclass_dataset(P, train_set, classes=[P.normal_label], count=P.main_count)
        test_set = get_subclass_dataset(P, test_set, classes=[P.normal_label])
        
    return train_set, test_set, full_test_set

def create_data_loaders(train_set, test_set, P):
    kwargs = {'pin_memory': False, 'num_workers': 4}
    train_loader = DataLoader(train_set, shuffle=True, batch_size=P.batch_size, **kwargs)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=P.test_batch_size, **kwargs)
    
    return train_loader, test_loader

def print_dataset_info(train_set, test_set, train_loader, test_loader):
    print("Number of normal test set:", len(test_set))
    print("Number of normal train set:", len(train_set))
    print("Length of train_set:", len(train_set))
    print("Length of test_set:", len(test_set))
    print("Unique labels (test_loader):", get_loader_unique_label(test_loader))
    print("Unique labels (train_loader):", get_loader_unique_label(train_loader))
    print("Number of train loader batches:", len(train_loader))

def prepare_ood_datasets(P, full_test_set, kwargs):
    ood_test_loader = {}
    
    for ood in P.ood_dataset:
        ood_test_set = get_subclass_dataset(P, full_test_set, classes=ood)
        ood_name = f'one_class_{ood}'
        print(f"Testset anomaly (class {ood_name}):", len(ood_test_set))
        ood_test_loader[ood_name] = DataLoader(ood_test_set, shuffle=False, batch_size=P.test_batch_size, **kwargs)
        print("Unique labels (ood_test_loader):", get_loader_unique_label(ood_test_loader[ood_name]))
        
    return ood_test_loader

def initialize_model(P, device):
    simclr_aug = C.get_simclr_augmentation(P, image_size=P.image_size).to(device)
    model = C.get_classifier(
        P.model, 
        n_classes=P.n_classes, 
        activation=P.activation_function, 
        mean=P.noise_mean, 
        std=P.noise_std, 
        noise_scale=P.noise_scale, 
        noist_probability=P.noist_probability
    ).to(device)
    model = C.get_shift_classifer(model, 2).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    
    if P.load_path is not None:
        checkpoint = torch.load(P.load_path)
        model.load_state_dict(checkpoint, strict=not P.no_strict)
        
    model.eval()
    
    return model, simclr_aug, criterion

def calculate_mean_auroc(auroc_dict):
    mean_auroc = sum(auroc_dict.values()) / len(auroc_dict)
    auroc_dict['one_class_mean'] = mean_auroc

def print_auroc_scores(auroc_dict, print_score):
    for ood, auroc in auroc_dict.items():
        message = f'[{ood} {auroc:.4f}] '
        if print_score:
            print(message)

def main():
    P = parse_args()

    cls_list = get_superclass_list(P.dataset)
    anomaly_labels = [elem for elem in cls_list if elem not in [P.normal_label]]
    
    P.n_gpus = torch.cuda.device_count()
    assert P.n_gpus <= 1  # no multi GPU
    P.multi_gpu = False
    
    device = setup_device(P.local_rank)
    
    if P.dataset == 'imagenet':
        P.batch_size = 1
        P.test_batch_size = 1
    
    train_set, test_set, full_test_set = initialize_datasets(P)
    train_loader, test_loader = create_data_loaders(train_set, test_set, P)
    
    print_dataset_info(train_set, test_set, train_loader, test_loader)
    
    P.ood_dataset = anomaly_labels
    if P.dataset in {
        'cifar10-versus-other-eval', 
        'cifar100-versus-other-eval', 
        'ISIC2018', 
        "mvtecad", 
        'cifar10-versus-100', 
        'cifar100-versus-10'
    }:
        P.ood_dataset = [1]
    
    print("P.ood_dataset", P.ood_dataset)
    
    ood_test_loader = prepare_ood_datasets(P, full_test_set, kwargs={'pin_memory': False, 'num_workers': 4})
    
    model, simclr_aug, criterion = initialize_model(P, device)
    
    with torch.no_grad():
        auroc_dict = eval_ood_detection(P, model, test_loader, ood_test_loader, train_loader=train_loader, simclr_aug=simclr_aug)
    
    if P.one_class_idx is not None:
        calculate_mean_auroc(auroc_dict)
    
    print_auroc_scores(auroc_dict, P.print_score)

if __name__ == "__main__":
    main()
