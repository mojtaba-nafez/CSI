import os
import gc
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import models.transform_layers as TL
#from utils import set_random_seed, normalize, get_auroc
from utils.utils import set_random_seed, normalize, get_auroc

# from adv_evaluation.pgd import PGD
from evals.pgd import PGD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hflip = TL.HorizontalFlipLayer().to(device)


def make_model_gradient(model, action):
    for param in model.parameters():
        param.requires_grad = action


class DifferentiableScoreModel(nn.Module):

    def __init__(self, P, device, model, simclr_aug):
        super(DifferentiableScoreModel, self).__init__()
        print(P)
        self.P = P
        self.device = device
        self.model = model
        self.simclr_aug = simclr_aug

    def get_scores(self, feats_dict, x):
        P = self.P
        device = self.device
        # convert to gpu tensor
        feats_sim = feats_dict['simclr'].to(device)
        feats_shi = feats_dict['shift'].to(device)
        N = feats_sim.size(0)

        # compute scores
        scores = []
        for f_sim, f_shi in zip(feats_sim, feats_shi):
            f_sim = [f.mean(dim=0, keepdim=True).requires_grad_() for f in f_sim.chunk(P.K_shift)]  # list of (1, d)
            f_shi = [f.mean(dim=0, keepdim=True).requires_grad_() for f in f_shi.chunk(P.K_shift)]  # list of (1, 4)
            score = torch.zeros(1, requires_grad=True).to(device)
            for shi in range(P.K_shift):
                # score = score + (f_sim[shi] * P.axis[shi].to(device)).sum(dim=1).requires_grad_().max() * torch.tensor(
                #    P.weight_sim[shi], requires_grad=True).to(device)
                score = score + (f_shi[shi][:, shi]) * torch.tensor(P.weight_shi[shi], requires_grad=True).to(device)

            score = score / P.K_shift
            scores.append(score)
        scores = torch.stack(scores).reshape(-1)

        assert scores.dim() == 1 and scores.size(0) == N  # (N)
        return scores.cpu()

    def get_features(self, data_name, model, data_batch,
                     simclr_aug=None, sample_num=1, layers=('simclr', 'shift')):
        P = self.P
        if not isinstance(layers, (list, tuple)):
            layers = [layers]

        # load pre-computed features if exists
        feats_dict = dict()

        # pre-compute features and save to the path
        left = [layer for layer in layers if layer not in feats_dict.keys()]
        if len(left) > 0:
            _feats_dict = self._get_features(model, data_batch,
                                             simclr_aug, sample_num, layers=left)

            for layer, feats in _feats_dict.items():
                feats_dict[layer] = feats

        return feats_dict

    def _get_features(self, model, data_batch, simclr_aug=None,
                      sample_num=1, layers=('simclr', 'shift')):
        P = self.P
        device = self.device
        if not isinstance(layers, (list, tuple)):
            layers = [layers]
        # check if arguments are valid
        assert simclr_aug is not None
        # compute features in full dataset
        model.eval()
        feats_all = {layer: [] for layer in layers}  # initialize: empty list
        x = data_batch.to(device)  # gpu tensor
        # compute features in one batch
        feats_batch = {layer: [] for layer in layers}  # initialize: empty list
        for seed in range(sample_num):
            set_random_seed(seed)
            if P.K_shift > 1:
                x_t = torch.cat([P.shift_trans(hflip(x), k) for k in range(P.K_shift)])
            else:
                x_t = x  # No shifting: SimCLR
            # x_t = simclr_aug(x_t)
            kwargs = {layer: True for layer in layers}  # only forward selected layers
            _, output_aux = model(x_t, **kwargs)
            # add features in one batch
            for layer in layers:
                feats = output_aux[layer].cpu()
                feats_batch[layer] += feats.chunk(P.K_shift)  # (B, d) cpu tensor

        # concatenate features in one batch
        for key, val in feats_batch.items():
            feats_batch[key] = torch.stack(val, dim=1)  # (B, T, d)

        # add features in full dataset
        for layer in layers:
            feats_all[layer] += [feats_batch[layer]]

        # concatenate features in full dataset
        for key, val in feats_all.items():
            feats_all[key] = torch.cat(val, dim=0)  # (N, T, d)

        # reshape order

        for key, val in feats_all.items():
            N, T, d = val.size()  # T = K * T'
            val = val.view(N, -1, P.K_shift, d)  # (N, T', K, d)
            val = val.transpose(2, 1)  # (N, 4, T', d)
            val = val.reshape(N, T, d)  # (N, T, d)
            feats_all[key] = val

        return feats_all

    def forward(self, x):
        P = self.P
        simclr_aug = self.simclr_aug
        kwargs = {
            'simclr_aug': simclr_aug,
            'sample_num': P.ood_samples,
            'layers': P.ood_layer,
        }

        P = self.P
        device = self.device
        with torch.set_grad_enabled(True):
            feats = self.get_features(P.dataset, self.model, x, **kwargs)  # (N, T, d)
            # print(feats['simclr'].shape)
            # print(x.shape)
            # print(feats['shift'].shape) # (100, 10, 2)
            
            # scores = self.get_scores(feats, x)
            output = feats['shift'].mean(dim=1, keepdim=True).requires_grad_()
            # print("output.shape", output.shape) # (100, 1, 2)
        #return scores
        return output

def eval_ood_detection(P, model, id_loader, ood_loaders, ood_scores, train_loader=None, simclr_aug=None):

    P.K_shift = 1
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
    auroc_dict = dict()
    for ood in ood_loaders.keys():
        auroc_dict[ood] = dict()

    assert len(ood_scores) == 1  # assume single ood_score for simplicity
    ood_score = ood_scores[0]

    base_path = os.path.split(P.load_path)[0]  # checkpoint directory

    kwargs = {
        'simclr_aug': simclr_aug,
        'sample_num': P.ood_samples,
        'layers': P.ood_layer,
    }
    if P.print_score:
        print('Pre-compute global statistics...')
    feats_train = get_features(P, f'{P.dataset}_train', model, train_loader, **kwargs)  # (M, T, d)

    P.axis = []
    for f in feats_train['simclr'].chunk(P.K_shift, dim=1):
        axis = f.mean(dim=1)  # (M, d)
        P.axis.append(normalize(axis, dim=1).to(device))
    if P.print_score:
        print('axis size: ' + ' '.join(map(lambda x: str(len(x)), P.axis)))

    f_sim = [f.mean(dim=1) for f in feats_train['simclr'].chunk(P.K_shift, dim=1)]  # list of (M, d)
    f_shi = [f.mean(dim=1) for f in feats_train['shift'].chunk(P.K_shift, dim=1)]  # list of (M, 4)

    weight_sim = []
    weight_shi = []

    for shi in range(P.K_shift):
        sim_norm = f_sim[shi].norm(dim=1)  # (M)
        shi_mean = f_shi[shi][:, shi]  # (M)
        weight_sim.append(1 / sim_norm.mean().item())
        weight_shi.append(1 / shi_mean.mean().item())

    P.weight_sim = weight_sim
    P.weight_shi = weight_shi
  
    if P.print_score:
        print(f'weight_sim:\t' + '\t'.join(map('{:.4f}'.format, P.weight_sim)))
        print(f'weight_shi:\t' + '\t'.join(map('{:.4f}'.format, P.weight_shi)))

    ## Preprocessing is Ended
    P, device, model, simclr_aug
    score_model = DifferentiableScoreModel(P, device, model, simclr_aug)
    P.attack = {'PGD': PGD(score_model, steps=P.steps, eps=P.eps, alpha=P.alpha)
                }[P.desired_attack]

    if P.print_score:
        print('Pre-compute features...')
    feats_id = get_features(P, P.dataset, model, id_loader, attack=P.in_attack, is_ood=False, **kwargs)  # (N, T, d)
    feats_ood = dict()
    for ood, ood_loader in ood_loaders.items():
        feats_ood[ood] = get_features(P, ood, model, ood_loader, attack=P.out_attack, is_ood=True, **kwargs)

    if P.print_score:
        print(f'Compute OOD scores... (score: {ood_score})')
    scores_id = get_scores(P, feats_id, ood_score).numpy()
    scores_ood = dict()
    if P.one_class_idx is not None:
        one_class_score = []

    for ood, feats in feats_ood.items():
        scores_ood[ood] = get_scores(P, feats, ood_score).numpy()
        auroc_dict[ood][ood_score] = get_auroc(scores_id, scores_ood[ood])
        if P.one_class_idx is not None:
            one_class_score.append(scores_ood[ood])

    if P.one_class_idx is not None:
        one_class_score = np.concatenate(one_class_score)
        one_class_total = get_auroc(scores_id, one_class_score)
        if P.print_score:
            print(f'One_class_real_mean: {one_class_total}')
        else:
            print(one_class_total)

    if P.print_score:
        print_score(P.dataset, scores_id)
        for ood, scores in scores_ood.items():
            print_score(ood, scores)

    return auroc_dict


def get_scores(P, feats_dict, ood_score):
    # convert to gpu tensor
    feats_sim = feats_dict['simclr'].to(device)
    feats_shi = feats_dict['shift'].to(device)
    N = feats_sim.size(0)

    # compute scores
    scores = []
    for f_sim, f_shi in zip(feats_sim, feats_shi):
        f_sim = [f.mean(dim=0, keepdim=True) for f in f_sim.chunk(P.K_shift)]  # list of (1, d)
        f_shi = [f.mean(dim=0, keepdim=True) for f in f_shi.chunk(P.K_shift)]  # list of (1, 4)
        score = 0
        for shi in range(P.K_shift):
            # score += (f_sim[shi] * P.axis[shi]).sum(dim=1).max().item() * P.weight_sim[shi]
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
    feats_dict = dict()
    # pre-compute features and save to the path
    left = [layer for layer in layers if layer not in feats_dict.keys()]
    if len(left) > 0:
        _feats_dict = _get_features(P, model, loader, P.dataset == 'imagenet',
                                    simclr_aug, sample_num, layers=left, attack=attack, is_ood=is_ood)
        for layer, feats in _feats_dict.items():
            feats_dict[layer] = feats  # update value
    return feats_dict


def _get_features(P, model, loader, imagenet=False, simclr_aug=None,
                  sample_num=1, layers=('simclr', 'shift'), attack=False, is_ood=False):
    if not isinstance(layers, (list, tuple)):
        layers = [layers]
    # check if arguments are valid
    assert simclr_aug is not None
    if imagenet is True:  # assume batch_size = 1 for ImageNet
        sample_num = 1
    model.eval()
    feats_all = {layer: [] for layer in layers}  # initialize: empty list

    for i, (x, targets) in enumerate(loader):
        if imagenet is True:
            x = torch.cat(x[0], dim=0)  # augmented list of x
        
        x = x.to(device)  # gpu tensor
        # compute features in one batch
        feats_batch = {layer: [] for layer in layers}  # initialize: empty list
        for seed in range(sample_num):
            set_random_seed(seed)
            if P.K_shift > 1:
                x_t = torch.cat([P.shift_trans(hflip(x), k) for k in range(P.K_shift)])
            else:
                x_t = x  # No shifting: SimCLR
            # x_t = simclr_aug(x_t)
            # compute augmented features
            with torch.no_grad():
                kwargs = {layer: True for layer in layers}  # only forward selected layers
                _, output_aux = model(x_t, **kwargs)
                # add features in one batch
            for layer in layers:
                feats = output_aux[layer].cpu()
                if imagenet is False:
                    feats_batch[layer] += feats.chunk(P.K_shift)
                else:
                    feats_batch[layer] += [feats]  # (B, d) cpu tensor
        # concatenate features in one batch
        for key, val in feats_batch.items():
            if imagenet:
                feats_batch[key] = torch.stack(val, dim=0)  # (B, T, d)
            else:
                feats_batch[key] = torch.stack(val, dim=1)  # (B, T, d)
        # add features in full dataset
        for layer in layers:
            feats_all[layer] += [feats_batch[layer]]

    # concatenate features in full dataset
    for key, val in feats_all.items():
        feats_all[key] = torch.cat(val, dim=0)  # (N, T, d)

    # reshape order
    if imagenet is False:
        # Convert [1,2,3,4, 1,2,3,4] -> [1,1, 2,2, 3,3, 4,4]
        for key, val in feats_all.items():
            N, T, d = val.size()  # T = K * T'
            val = val.view(N, -1, P.K_shift, d)  # (N, T', K, d)
            val = val.transpose(2, 1)  # (N, 4, T', d)
            val = val.reshape(N, T, d)  # (N, T, d)
            feats_all[key] = val

    return feats_all


def print_score(data_name, scores):
    quantile = np.quantile(scores, np.arange(0, 1.1, 0.1))
    print('{:18s} '.format(data_name) +
          '{:.4f} +- {:.4f}    '.format(np.mean(scores), np.std(scores)) +
          '    '.join(['q{:d}: {:.4f}'.format(i * 10, quantile[i]) for i in range(11)]))
