# ./common/swa.py

import torch
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.swa_utils import update_bn

def initialize_swa_model(model):
    """
    Initialize the SWA model from the given model.
    """
    return AveragedModel(model)

def get_swa_scheduler(optimizer, swa_lr):
    """
    Get the SWA learning rate scheduler.
    """
    return SWALR(optimizer, swa_lr=swa_lr)

def update_swa_model(swa_model, model, epoch, start_swa_epoch, swa_update_frequency, swa_scheduler):
    """
    Update the SWA model weights based on the given conditions.
    """
    if epoch > start_swa_epoch and (epoch - start_swa_epoch) % swa_update_frequency == 0:
        swa_model.update_parameters(model)
        swa_scheduler.step()

def update_batch_norm(train_loader, swa_model):
    """
    Update batch normalization layers for the SWA model.
    """
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    swa_model.cuda()
    update_bn(train_loader, swa_model, device)
