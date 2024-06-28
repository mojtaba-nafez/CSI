import os
import gc

from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import models.transform_layers as TL
from utils.utils import set_random_seed, normalize, get_auroc
# from evals.evals import get_auroc


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hflip = TL.HorizontalFlipLayer().to(device)


def make_model_gradient(model, action):
    for param in model.parameters():
        param.requires_grad = action

def eval_ood_detection(P, model, id_loader, ood_loaders, ood_scores, train_loader=None, simclr_aug=None):
    P.K_shift = 1
    P.PGD_constant = 2.5
    P.alpha = (P.PGD_constant * P.eps) / P.steps
    
    


    auroc_dict = dict()
    for ood in ood_loaders.keys():
        auroc_dict[ood] = dict()

    assert len(ood_scores) == 1  # assume single ood_score for simplicity
    ood_score = ood_scores[0]

    kwargs = {
        'simclr_aug': simclr_aug,
        'sample_num': P.ood_samples,
        'layers': P.ood_layer,
    }

    print('Pre-compute global statistics...')
    # feats_train["simclr"]:   torch.Size([5000, 40, 128])
    # feats_train["shift"]:    torch.Size([5000, 40, 4])
    feats_train = get_features(P, f'{P.dataset}_train', model, train_loader, **kwargs)  # (M, T, d)
    P.axis = []
    for f in feats_train['simclr'].chunk(P.K_shift, dim=1):
        # f.shape: torch.Size([5000, 10, 128])
        # f.mean(dim=1).shape: torch.Size([5000, 128])
        axis = f.mean(dim=1)  # (M, d)
        P.axis.append(normalize(axis, dim=1).to(device))
    # P.axis: [torch.Size([5000, 128]), torch.Size([5000, 128]), torch.Size([5000, 128]), torch.Size([5000, 128])]
    print('axis size: ' + ' '.join(map(lambda x: str(len(x)), P.axis)))

    # f_shi: [torch.Size([5000, 128]), torch.Size([5000, 128]), torch.Size([5000, 128]), torch.Size([5000, 128])]
    f_sim = [f.mean(dim=1) for f in feats_train['simclr'].chunk(P.K_shift, dim=1)]  # list of (M, d)

    # f_shi: [torch.Size([5000, 4]), torch.Size([5000, 4]), torch.Size([5000, 4]), torch.Size([5000, 4])]
    f_shi = [f.mean(dim=1) for f in feats_train['shift'].chunk(P.K_shift, dim=1)]  # list of (M, 4)

    weight_sim = []
    weight_shi = []
    for shi in range(P.K_shift):
        # sim_norm = torch.Size([5000])
        sim_norm = f_sim[shi].norm(dim=1)  # (M)
        # shi_mean = torch.Size([5000])
        shi_mean = f_shi[shi][:, shi]  # (M)
        weight_sim.append(1 / sim_norm.mean().item())
        weight_shi.append(1 / shi_mean.mean().item())
    # weight_shi= (4,)
    # weight_shi= (4,)
    # ood_score == 'CSI'
    if ood_score == 'simclr':
        P.weight_sim = [1]
        P.weight_shi = [0]
    elif ood_score == 'CSI':
        P.weight_sim = weight_sim
        P.weight_shi = weight_shi
    else:
        raise ValueError()

    print(f'weight_sim:\t' + '\t'.join(map('{:.4f}'.format, P.weight_sim)))
    print(f'weight_shi:\t' + '\t'.join(map('{:.4f}'.format, P.weight_shi)))
    
    ## Preprocessing is Ended
    P, device, model, simclr_aug
    
    print('Pre-compute features...')
    # feats_id["shift"].shape = torch.Size([1000, 40, 4])
    # feats_id["simclr"] = torch.Size([1000, 40, 128])
    feats_id = get_features(P, P.dataset, model, id_loader, attack=False, is_ood=False, **kwargs)  # (N, T, d)
    feats_ood = dict()
    # ood_loaders.items() = [1,2,3,4,5,6,7,8,9]
    for ood, ood_loader in ood_loaders.items():
        # feats_ood[ood]["shift"]: torch.Size([1000, 40, 4])
        # feats_ood[ood]["simclr"]: torch.Size([1000, 40, 128])
        feats_ood[ood] = get_features(P, ood, model, ood_loader, attack=False, is_ood=True, **kwargs)

    print(f'Compute OOD scores... (score: {ood_score})')
    # scores_id.shape=(1000,)
    scores_id = get_scores(P, feats_id, ood_score).numpy()

    scores_ood = dict()
    if P.one_class_idx is not None:
        one_class_score = []

    for ood, feats in feats_ood.items():
        # scores_ood[ood].shape=(1000,)
        scores_ood[ood] = get_scores(P, feats, ood_score).numpy()
        # auroc_dict[ood][ood_score]: a float(0<=  <=1)
        auroc_dict[ood][ood_score] = get_auroc(scores_id, scores_ood[ood])
        if P.one_class_idx is not None:
            one_class_score.append(scores_ood[ood])

    if P.one_class_idx is not None:
        one_class_score = np.concatenate(one_class_score)
        one_class_total = get_auroc(scores_id, one_class_score)
        print(f'One_class_real_mean: {one_class_total}')

    if P.print_score:
        print_score(P.dataset, scores_id)
        for ood, scores in scores_ood.items():
            print_score(ood, scores)

    return auroc_dict


def get_scores(P, feats_dict, ood_score):
    # convert to gpu tensor
    # feats_shi: [1000, 40, 128]
    feats_sim = feats_dict['simclr'].to(device)
    # feats_shi: [1000, 40, 4]
    feats_shi = feats_dict['shift'].to(device)
    N = feats_sim.size(0)

    # compute scores
    scores = []

    # f_shi.shape: torch.Size([40, 4])
    # f_sim.shape: torch.Size([40, 128])
    for f_sim, f_shi in zip(feats_sim, feats_shi):

        # f_sim: 4 * torch.Size([1, 128])
        f_sim = [f.mean(dim=0, keepdim=True) for f in f_sim.chunk(P.K_shift)]  # list of (1, d)
        # f_shi: 4 * torch.Size([1, 4])
        f_shi = [f.mean(dim=0, keepdim=True) for f in f_shi.chunk(P.K_shift)]  # list of (1, 4)
        score = 0
        for shi in range(P.K_shift):
            # (f_sim[shi].shape, P.axis[shi].shape) = ( torch.Size([1, 128]), torch.Size([5000, 128]) )
            # (f_sim[shi] * P.axis[shi]).shape = torch.Size([5000, 128])
            # (f_sim[shi] * P.axis[shi]).sum(dim=1) = torch.Size([5000])
            score += (f_sim[shi] * P.axis[shi]).sum(dim=1).max().item() * P.weight_sim[shi]
            score += f_shi[shi][:, shi].item() * P.weight_shi[shi]
        score = score / P.K_shift
        scores.append(score)
    scores = torch.tensor(scores)

    assert scores.dim() == 1 and scores.size(0) == N  # (N)
    return scores.cpu()


def get_features(P, data_name, model, loader,
                 simclr_aug=None, sample_num=1, layers=('simclr', 'shift'), attack=False, is_ood=False):

    if not isinstance(layers, (list, tuple)):
        layers = [layers]

    # load pre-computed features if exists
    feats_dict = dict()
    # for layer in layers:
    #     path = prefix + f'_{data_name}_{layer}.pth'
    #     if os.path.exists(path):
    #         feats_dict[layer] = torch.load(path)

    # pre-compute features and save to the path
    # left= ['simclr', 'shift']
    left = [layer for layer in layers if layer not in feats_dict.keys()]
    if len(left) > 0:
        _feats_dict = _get_features(P, model, loader, P.dataset == 'imagenet',
                                    simclr_aug, sample_num, layers=left, attack=attack, is_ood=is_ood)

        for layer, feats in _feats_dict.items():
            feats_dict[layer] = feats  # update value

    return feats_dict


def _get_features(P, model, loader, imagenet=False, simclr_aug=None,
                  sample_num=1, layers=('simclr', 'shift'), attack=False, is_ood=False):
    # layers = ['simclr', 'shift']
    if not isinstance(layers, (list, tuple)):
        layers = [layers]

    # check if arguments are valid
    assert simclr_aug is not None

    if imagenet is True:  # assume batch_size = 1 for ImageNet
        sample_num = 1

    # compute features in full dataset
    model.eval()
    feats_all = {layer: [] for layer in layers}  # initialize: empty list
    for i, (x, _) in enumerate(loader):
        if imagenet is True:
            x = torch.cat(x[0], dim=0)  # augmented list of x
        x = x.to(device)  # gpu tensor

        # compute features in one batch
        feats_batch = {layer: [] for layer in layers}  # initialize: empty list
        # sample_num=10
        for seed in range(sample_num):
            set_random_seed(seed)

            if P.K_shift > 1:
                # train time call:
                #   x   = torch.Size([128, 3, 32, 32])
                #   x_t = torch.Size([512, 3, 32, 32])
                # test time call:
                #   x   = torch.Size([100, 3, 32, 32])
                #   x_t = torch.Size([400, 3, 32, 32])
                x_t = torch.cat([P.shift_trans(hflip(x), k) for k in range(P.K_shift)])
            else:
                x_t = x # No shifting: SimCLR
            x_t = simclr_aug(x_t)

            # compute augmented features
            with torch.no_grad():
                # layers = ['simclr', 'shift']
                kwargs = {layer: True for layer in layers}  # only forward selected layers
                _, output_aux = model(x_t, **kwargs)

            # add features in one batch
            for layer in layers:
                # train time call
                #   output_aux["shift"] torch.Size([512, 4])
                #   output_aux["simclr"] torch.Size([512, 128])
                # test time call
                #   output_aux["shift"] torch.Size([400, 4])
                #   output_aux["simclr"] torch.Size([400, 128])
                feats = output_aux[layer].cpu()
                # imagenet = False
                if imagenet is False:
                    # feats.chunk(P.K_shift):
                    #   Train:   4 * torch.Size([128, 128])   ||   4 * torch.Size([128, 4])
                    #   Test:    4 * torch.Size([100, 128])   ||   4 * torch.Size([100, 4])

                    # feats_batch[layer] = array of len=4
                    # train:    40 * torch.Size([128, 128])  ||  40 * torch.Size([128, 4])
                    # test:     40 * torch.Size([100, 128])  ||  40 * torch.Size([100, 4])
                    feats_batch[layer] += feats.chunk(P.K_shift)
                else:
                    feats_batch[layer] += [feats]  # (B, d) cpu tensor

        # concatenate features in one batch
        for key, val in feats_batch.items():
            if imagenet:
                feats_batch[key] = torch.stack(val, dim=0)  # (B, T, d)
            else:
                feats_batch[key] = torch.stack(val, dim=1)  # (B, T, d)
        # feats_batch
        # feats_batch["simclr"] = torch.Size([128, 40, 128]) || torch.Size([100, 40, 128])
        # feats_batch["shift"]  = torch.Size([128, 40, 4])   || torch.Size([100, 40, 4])

        # add features in full dataset
        for layer in layers:
            feats_all[layer] += [feats_batch[layer]]
    # feats_all["shift or simclr"] is an array len=40 --> element: [128, 40, 128]
    #   train time call:   simclr=[40, 128, 40, 128]  || shift=[40, 128, 40, 4]
    #   test time call:    simclr=[10, 100, 40, 128]  || shift=[10, 100, 40, 4]

    # concatenate features in full dataset
    for key, val in feats_all.items():
        feats_all[key] = torch.cat(val, dim=0)  # (N, T, d)
    # train time call feats_all[key]:
    #        torch.Size([5000, 40, 128])
    #        torch.Size([5000, 40, 4])
    # train time call feats_all[key]:
    #        torch.Size([1000, 40, 128])
    #        torch.Size([1000, 40, 4])

    # reshape order
    if imagenet is False:
        # Convert [1,2,3,4, 1,2,3,4] -> [1,1, 2,2, 3,3, 4,4]
        for key, val in feats_all.items():
            N, T, d = val.size()  # T = K * T'
            val = val.view(N, -1, P.K_shift, d)  # (N, T', K, d)
            val = val.transpose(2, 1)  # (N, 4, T', d)
            val = val.reshape(N, T, d)  # (N, T, d)
            feats_all[key] = val
    # train time call feats_all[key]:
    #        torch.Size([5000, 40, 128])
    #        torch.Size([5000, 40, 4])
    # train time call feats_all[key]:
    #        torch.Size([1000, 40, 128])
    #        torch.Size([1000, 40, 4])
    return feats_all


def print_score(data_name, scores):
    quantile = np.quantile(scores, np.arange(0, 1.1, 0.1))
    print('{:18s} '.format(data_name) +
          '{:.4f} +- {:.4f}    '.format(np.mean(scores), np.std(scores)) +
          '    '.join(['q{:d}: {:.4f}'.format(i * 10, quantile[i]) for i in range(11)]))

