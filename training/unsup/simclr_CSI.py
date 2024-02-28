import time

import torch
import torch.optim
import models.transform_layers as TL
from training.contrastive_loss import get_similarity_matrix, NT_xent
from utils.utils import AverageMeter, normalize
from utils.virtual_outliers import create_faiss_index, find_boundary_points, generate_candidate_outliers, select_best_outliers, synthesize_outliers
from torch.utils.data import DataLoader, TensorDataset
from utils.utils import AverageMeter, normalize

device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
hflip = TL.HorizontalFlipLayer().to(device)


def train(P, epoch, model, criterion, optimizer, scheduler, loader, train_exposure_loader=None, logger=None,
          simclr_aug=None, linear=None, linear_optim=None):

    assert simclr_aug is not None
    assert P.sim_lambda == 1.0  # to avoid mistake
    P.K_shift = 2
    assert P.K_shift > 1

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = dict()
    losses['cls'] = AverageMeter()
    losses['sim'] = AverageMeter()
    losses['shift'] = AverageMeter()

    check = time.time()
    
    train_exposure_loader_iterator = iter(train_exposure_loader)

    if P.virtual_outliers and epoch > P.warmup_epoch:
        all_embeddings = []
        model.eval()
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device)
                features = model.penultimate(images).detach()
                all_embeddings.append(features.detach().cpu())
        all_embeddings = torch.cat(all_embeddings, dim=0)
        virtual_outliers_embeddings = synthesize_outliers(all_embeddings.to(device), all_embeddings.shape[0], P.K, P.num_boundary_points, P.num_candidate_outliers, P.std_dev).detach()
        virtual_outliers_loader = DataLoader(TensorDataset(virtual_outliers_embeddings), batch_size=P.batch_size, shuffle=True)
        virtual_outliers_loader_iterator = iter(virtual_outliers_loader)
        model.train()
    else:
        train_exposure_loader_iterator = iter(train_exposure_loader)
        
    print("len(train_exposure_loader_iterator), len(loader): ", len(train_exposure_loader_iterator), len(loader))
    print("cl_no_hflip=", P.cl_no_hflip)
    
    for n, (images, labels) in enumerate(loader):
        
        # print(exposure_images.shape, images.shape, labels.shape)
        model.train()
        count = n * P.n_gpus  # number of trained samples

        data_time.update(time.time() - check)
        check = time.time()
        
        if P.virtual_outliers and epoch > P.warmup_epoch:
            try:
                exposure_embeddings = next(virtual_outliers_loader_iterator)
            except StopIteration:
                virtual_outliers_loader_iterator = iter(virtual_outliers_loader)
                exposure_embeddings = next(virtual_outliers_loader_iterator)
            exposure_embeddings = exposure_embeddings.to(device)
            
            batch_size = images.size(0)
            images = images.to(device)
            
            if P.cl_no_hflip:
                images1, images2 = images.repeat(2, 1, 1, 1).chunk(2)  # hflip
            else:
                images1, images2 = hflip(images.repeat(2, 1, 1, 1)).chunk(2)  # hflip

            features = torch.cat([model.penultimate(images1), exposure_embeddings, model.penultimate(images2), exposure_embeddings]) # inlier / exposure / augmented-inlier / exposure 
            
        else:
            try:
                exposure_images, _ = next(train_exposure_loader_iterator)
            except StopIteration:
                train_exposure_loader_iterator = iter(train_exposure_loader)
                exposure_images, _ = next(train_exposure_loader_iterator)
        
            ### SimCLR loss ###
            if P.dataset != 'imagenet':
                batch_size = images.size(0)
                images = images.to(device)
                exposure_images = exposure_images.to(device)
                if P.cl_no_hflip:
                    images1, images2 = images.repeat(2, 1, 1, 1).chunk(2)  # hflip
                else:
                    images1, images2 = hflip(images.repeat(2, 1, 1, 1)).chunk(2)  # hflip
                exposure_images1, exposure_images2 = hflip(exposure_images.repeat(2, 1, 1, 1)).chunk(2)  # hflip
            else:
                batch_size = images[0].size(0)
                images1, images2 = images[0].to(device), images[1].to(device)
            
            images1 = torch.cat([images1, exposure_images1])
            images2 = torch.cat([images2, exposure_images2])
            
            images_pair = torch.cat([images1, images2], dim=0)  # 8B
            images_pair = simclr_aug(images_pair)  # transform
        
            features = model.penultimate(images_pair)        

        labels = labels.to(device)
        shift_labels = torch.cat([torch.ones_like(labels), torch.zeros_like(labels)], 0)  # B -> 4B
        shift_labels = shift_labels.repeat(2)
        
        simclr = normalize(model.simclr_layer(features))  # normalize
        sim_matrix = get_similarity_matrix(simclr, multi_gpu=P.multi_gpu)
        loss_sim = NT_xent(sim_matrix, temperature=0.5) * P.sim_lambda

        loss_shift = criterion(model.shift_cls_layer(features), shift_labels)

        ### total loss ###
        loss = loss_sim + loss_shift

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step(epoch - 1 + n / len(loader))
        lr = optimizer.param_groups[0]['lr']

        batch_time.update(time.time() - check)

        losses['cls'].update(0, batch_size)
        losses['sim'].update(loss_sim.item(), batch_size)
        losses['shift'].update(loss_shift.item(), batch_size)

        if count % 50 == 0:
            log_('[Epoch %3d; %3d] [Time %.3f] [Data %.3f] [LR %.5f]\n'
                 '[LossC %f] [LossSim %f] [LossShift %f]' %
                 (epoch, count, batch_time.value, data_time.value, lr,
                  losses['cls'].value, losses['sim'].value, losses['shift'].value))

    log_('[DONE] [Time %.3f] [Data %.3f] [LossC %f] [LossSim %f] [LossShift %f]' %
         (batch_time.average, data_time.average,
          losses['cls'].average, losses['sim'].average, losses['shift'].average))

    if logger is not None:
        logger.scalar_summary('train/loss_cls', losses['cls'].average, epoch)
        logger.scalar_summary('train/loss_sim', losses['sim'].average, epoch)
        logger.scalar_summary('train/loss_shift', losses['shift'].average, epoch)
        logger.scalar_summary('train/batch_time', batch_time.average, epoch)

