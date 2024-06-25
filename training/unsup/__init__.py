def setup(mode, P):
    fname = f'{P.dataset}_{P.model}_unsup_{mode}'

    from .simclr_CSI import train
    fname += f'_shift_{P.shift_trans_type}'
    
    if P.one_class_idx is not None:
        fname += f'_one_class_{P.one_class_idx}'

    if P.suffix is not None:
        fname += f'_{P.suffix}'
    fname += str(P.normal_label)
    return train, fname

