import torch
import torch.nn as nn
from torch.utils.data import DataLoader

#from arguments import parse_args
import models.classifier as C
from datasets import get_dataset, get_superclass_list, get_subclass_dataset
from copy import deepcopy
from evaluation.eval import evaluate
from argparse import ArgumentParser


def parse_args(default=False):
    """Command-line argument parser for training."""

    parser = ArgumentParser(description='Pytorch implementation of COBRA')

    parser.add_argument('--dataset', help='Dataset',
                        choices=['fashion-mnist', 'mnist', 'cifar10', 'cifar100', 'imagenet', 'svhn-10'],
                        default="cifar10", type=str)
    parser.add_argument('--one_class_idx', help='None: multi-class, Not None: one-class',
                        default=None, type=int)
    parser.add_argument('--model', help='Model',
                        choices=['resnet18', 'resnet18_imagenet'], default="resnet18", type=str)
    parser.add_argument('--simclr_dim', help='Dimension of simclr layer',
                        default=128, type=int)

    parser.add_argument('--shift_trans_type', help='shifting transformation type', default='rotation',
                        choices=['rotation', 'cutperm', 'none'], type=str)
    parser.add_argument("--local_rank", type=int,
                        default=0, help='Local rank for distributed learning')
    parser.add_argument('--resume_path', help='Path to the resume checkpoint',
                        default=None, type=str)
    parser.add_argument('--load_path', help='Path to the loading checkpoint',
                        default="./cifar10_oc_class0.model", type=str)
    parser.add_argument("--no_strict", help='Do not strictly load state_dicts',
                        action='store_true')
    parser.add_argument('--suffix', help='Suffix for the log dir',
                        default=None, type=str)
    parser.add_argument('--error_step', help='Epoch steps to compute errors',
                        default=5, type=int)
    parser.add_argument('--save_step', help='Epoch steps to save models',
                        default=10, type=int)

    ##### Training Configurations #####
    parser.add_argument('--epochs', help='Epochs',
                        default=1000, type=int)
    parser.add_argument('--optimizer', help='Optimizer',
                        choices=['sgd', 'lars'],
                        default='lars', type=str)
    parser.add_argument('--lr_scheduler', help='Learning rate scheduler',
                        choices=['step_decay', 'cosine'],
                        default='cosine', type=str)
    parser.add_argument('--warmup', help='Warm-up epochs',
                        default=10, type=int)
    parser.add_argument('--lr_init', help='Initial learning rate',
                        default=1e-1, type=float)
    parser.add_argument('--weight_decay', help='Weight decay',
                        default=1e-6, type=float)
    parser.add_argument('--batch_size', help='Batch size',
                        default=64, type=int)
    parser.add_argument('--test_batch_size', help='Batch size for test loader',
                        default=64, type=int)

    ##### Objective Configurations #####
    parser.add_argument('--sim_lambda', help='Weight for SimCLR loss',
                        default=1.0, type=float)
    parser.add_argument('--temperature', help='Temperature for similarity',
                        default=0.5, type=float)

    ##### Evaluation Configurations #####
    parser.add_argument("--ood_dataset", help='Datasets for OOD detection',
                        default=None, nargs="*", type=str)
    parser.add_argument("--ood_score", help='score function for OOD detection',
                        default=['COBRA'], nargs="+", type=str)
    parser.add_argument("--ood_layer", help='layer for OOD scores',
                        choices=['penultimate', 'simclr', 'shift'],
                        default=['simclr', 'shift'], nargs="+", type=str)
    parser.add_argument("--ood_samples", help='number of samples to compute OOD score',
                        default=10, type=int)
    parser.add_argument("--ood_batch_size", help='batch size to compute OOD score',
                        default=100, type=int)
    parser.add_argument("--resize_factor", help='resize scale is sampled from [resize_factor, 1.0]',
                        default=0.08, type=float)
    parser.add_argument("--resize_fix", help='resize scale is fixed to resize_factor (not (resize_factor, 1.0])',
                        action='store_true')

    parser.add_argument("--print_score", default=True, help='print quantiles of ood score',
                        action='store_true')
    parser.add_argument("--save_score", help='save ood score for plotting histogram',
                        action='store_true')

    parser.add_argument('--attack_type', type=str, default='linf',
                        help='adversarial l_p')
    parser.add_argument('--epsilon', type=float, default=0.0314,
                        help='maximum perturbation of adversaries (8/255 for cifar-10)')
    parser.add_argument('--k', type=int, default=10,
                        help='maximum iteration when generating adversarial examples')
    parser.add_argument('--random_start', type=bool, default=True,
                        help='True for PGD')
    parser.add_argument('--regularize_to', default='other', type=str, help='original/other')
    parser.add_argument('--loss_type', type=str, default='sim', help='loss type for Rep')

    parser.add_argument('--min', type=float, default=0.0, help='min for cliping image')
    parser.add_argument('--max', type=float, default=1.0, help='max for cliping image')
    parser.add_argument('--lamda', default=256, type=float)
    parser.add_argument("--in_attack", help='save ood score for plotting histogram',
                        default=False, action='store_true')
    parser.add_argument("--out_attack", help='save ood score for plotting histogram',
                        default=False, action='store_true')

    parser.add_argument('--eps', type=float, default=0.0314,
                        help='maximum perturbation of adversaries (8/255 for cifar-10)')
    parser.add_argument('--steps', type=int, default=10,
                        help='maximum iteration when generating adversarial examples')
    if default:
        return parser.parse_args('')  # empty string
    else:
        return parser.parse_args()


def main():
    P = parse_args()

    ### Set torch device ###
    P.n_gpus = torch.cuda.device_count()
    P.multi_gpu = False
    if torch.cuda.is_available():
        torch.cuda.set_device(P.local_rank)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    ### Initialize dataset ###
    if P.dataset == 'imagenet':
        P.batch_size = 1
        P.test_batch_size = 1
    train_set, test_set, image_size, n_classes = get_dataset(P, dataset=P.dataset, eval=True)
    P.image_size = image_size
    P.n_classes = n_classes
    if P.one_class_idx is not None:
        cls_list = get_superclass_list(P.dataset)
        P.n_superclasses = len(cls_list)
        full_test_set = deepcopy(test_set)  # test set of full classes
        train_set = get_subclass_dataset(train_set, classes=cls_list[P.one_class_idx])
        test_set = get_subclass_dataset(test_set, classes=cls_list[P.one_class_idx])
    kwargs = {'pin_memory': False, 'num_workers': 4}
    train_loader = DataLoader(train_set, shuffle=True, batch_size=P.batch_size, **kwargs)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=P.test_batch_size, **kwargs)
    if P.ood_dataset is None:
        if P.one_class_idx is not None:
            P.ood_dataset = list(range(P.n_superclasses))
            P.ood_dataset.pop(P.one_class_idx)
        elif P.dataset == 'cifar10':
            P.ood_dataset = ['svhn', 'cifar100', 'mnist', 'imagenet', "fashion-mnist"]
    ood_test_loader = dict()
    for ood in P.ood_dataset:
        if P.one_class_idx is not None:
            ood_test_set = get_subclass_dataset(full_test_set, classes=cls_list[ood])
            ood = f'one_class_{ood}'
        else:
            ood_test_set = get_dataset(P, dataset=ood, test_only=True, image_size=P.image_size, eval=True)
        ood_test_loader[ood] = DataLoader(ood_test_set, shuffle=False, batch_size=P.test_batch_size, **kwargs)

    ### Initialize model ###
    simclr_aug = C.get_simclr_augmentation(P, image_size=P.image_size).to(device)
    P.shift_trans, P.K_shift = C.get_shift_module(P, eval=True)
    P.shift_trans = P.shift_trans.to(device)

    model = C.get_classifier(P.model, n_classes=P.n_classes).to(device)
    model = C.get_shift_classifer(model, P.K_shift).to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    if P.load_path is not None:
        checkpoint = torch.load(P.load_path)
        model.load_state_dict(checkpoint, strict=False)

    model.eval()

    P.desired_attack = "PGD"
    P.PGD_constant = 2.5
    P.alpha = (P.PGD_constant * P.eps) / P.steps
    print("Attack targets: ")
    if P.in_attack:
        print("- Normal")
    if P.out_attack:
        print("- Anomaly")

    if P.out_attack or P.in_attack:
        print("Desired Attack:", P.desired_attack)
        print("Epsilon:", P.eps)
        if P.desired_attack == 'PGD':
            print("Steps:", P.steps)

    auroc_dict = evaluate(P, model, test_loader, ood_test_loader, P.ood_score,
                                    train_loader=train_loader, simclr_aug=simclr_aug)

    if P.one_class_idx is not None:
        mean_dict = dict()
        for ood_score in P.ood_score:
            mean = 0
            for ood in auroc_dict.keys():
                mean += auroc_dict[ood][ood_score]
            mean_dict[ood_score] = mean / len(auroc_dict.keys())
        auroc_dict['one_class_mean'] = mean_dict

    bests = []
    for ood in auroc_dict.keys():
        message = ''
        best_auroc = 0
        for ood_score, auroc in auroc_dict[ood].items():
            message += '[%s %s %.4f] ' % (ood, ood_score, auroc)
            if auroc > best_auroc:
                best_auroc = auroc
        message += '[%s %s %.4f] ' % (ood, 'best', best_auroc)
        if P.print_score:
            print(message)
        bests.append(best_auroc)

    bests = map('{:.4f}'.format, bests)


if __name__ == '__main__':
    main()
