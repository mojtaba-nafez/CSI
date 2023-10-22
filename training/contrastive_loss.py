import torch
import torch.distributed as dist
import diffdist.functional as distops


def get_similarity_matrix(outputs, chunk=2, multi_gpu=False):
    '''
        Compute similarity matrix
        - outputs: (B', d) tensor for B' = B * chunk
        - sim_matrix: (B', B') tensor
    '''

    if multi_gpu:
        outputs_gathered = []
        for out in outputs.chunk(chunk):
            gather_t = [torch.empty_like(out) for _ in range(dist.get_world_size())]
            gather_t = torch.cat(distops.all_gather(gather_t, out))
            outputs_gathered.append(gather_t)
        outputs = torch.cat(outputs_gathered)

    sim_matrix = torch.mm(outputs, outputs.t())  # (B', d), (d, B') -> (B', B')

    return sim_matrix


def NT_xent(sim_matrix, temperature=0.5, chunk=2, eps=1e-8):
    '''
        Compute NT_xent loss
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    '''

    device = sim_matrix.device

    B = sim_matrix.size(0) // chunk  # B = B' / chunk

    eye = torch.eye(B * chunk).to(device)  # (B', B')
    sim_matrix = torch.exp(sim_matrix / temperature) * (1 - eye)  # remove diagonal

    denom = torch.sum(sim_matrix, dim=1, keepdim=True)
    sim_matrix = -torch.log(sim_matrix / (denom + eps) + eps)  # loss matrix

    loss = torch.sum(sim_matrix[:B, B:].diag() + sim_matrix[B:, :B].diag()) / (2 * B)

    return loss


def Supervised_NT_xent(sim_matrix, labels, temperature=0.5, chunk=2, eps=1e-8, multi_gpu=False):
    '''
        Compute NT_xent loss
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    '''

    device = sim_matrix.device

    if multi_gpu:
        gather_t = [torch.empty_like(labels) for _ in range(dist.get_world_size())]
        labels = torch.cat(distops.all_gather(gather_t, labels))
    labels = labels.repeat(2)

    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    sim_matrix = sim_matrix - logits_max.detach()

    B = sim_matrix.size(0) // chunk  # B = B' / chunk

    eye = torch.eye(B * chunk).to(device)  # (B', B')
    sim_matrix = torch.exp(sim_matrix / temperature) * (1 - eye)  # remove diagonal

    denom = torch.sum(sim_matrix, dim=1, keepdim=True)
    sim_matrix = -torch.log(sim_matrix / (denom + eps) + eps)  # loss matrix

    labels = labels.contiguous().view(-1, 1)
    Mask = torch.eq(labels, labels.t()).float().to(device)
    #Mask = eye * torch.stack([labels == labels[i] for i in range(labels.size(0))]).float().to(device)
    Mask = Mask / (Mask.sum(dim=1, keepdim=True) + eps)

    loss = torch.sum(Mask * sim_matrix) / (2 * B)

    return loss

def AnomalyContrastiveLoss(sim_matrix, temperature):
    # Validate the size of sim_matrix
    n = sim_matrix.shape[0] // 4
    assert sim_matrix.shape == (4 * n, 4 * n)
    
    image_sim = sim_matrix[:2 * n, :2 * n]
    exposure_sim = sim_matrix[2 * n:, 2 * n:]
    
    # For images: pull augmentations and other images, push exposures
    pos_image = torch.exp(image_sim / temperature)
    neg_image = torch.exp(image_sim[:, 2 * n:] / temperature)
    loss_image = -torch.log(pos_image / (pos_image + neg_image.sum(dim=-1, keepdim=True))).mean()
    
    # For exposures: pull only its augmentations, push images and other exposures
    pos_exposure = torch.exp(torch.diag(exposure_sim) / temperature)
    neg_exposure = torch.exp(exposure_sim / temperature) + torch.exp(exposure_sim[:, :2 * n] / temperature)
    torch.diagonal(neg_exposure, offset=0, dim1=0, dim2=1).fill_(0)  # exclude self similarity
    loss_exposure = -torch.log(pos_exposure / (pos_exposure + neg_exposure.sum(dim=-1))).mean()

    total_loss = loss_image + loss_exposure
    return total_loss

