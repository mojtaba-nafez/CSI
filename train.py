from utils.utils import Logger
from utils.utils import save_checkpoint
from utils.utils import save_linear_checkpoint
import os
from common.train import *
import time
from training.simclr_CSI import train

start_time = time.time()

fname = f'unode_{P.dataset}_{P.model}'
if P.normal_label is not None:
    fname += f'_one_class_{P.normal_label}'
if P.suffix is not None:
    fname += f'_{P.suffix}'

logger = Logger(fname, ask=not resume, local_rank=P.local_rank)
logger.log(P)
logger.log(model)

if P.multi_gpu:
    linear = model.module.linear
else:
    linear = model.linear
linear_optim = torch.optim.Adam(linear.parameters(), lr=1e-3, betas=(.9, .999), weight_decay=P.weight_decay)

epoch = 0
# Run experiments
for epoch in range(start_epoch, P.epochs + 1):
    if P.timer is not None and P.timer < (time.time() - start_time):
        break
    logger.log_dirname(f"Epoch {epoch}")
    model.train()

    if P.multi_gpu:
        train_sampler.set_epoch(epoch)

    kwargs = {}
    kwargs['linear'] = linear
    kwargs['linear_optim'] = linear_optim
    kwargs['simclr_aug'] = simclr_aug

    
    if epoch > P.unfreeze_pretrain_model_epoch:
        for param in model.parameters():
            param.requires_grad = True

    train(P, epoch, model, criterion, optimizer, scheduler_warmup, train_loader, logger=logger, **kwargs)

    model.eval()
    save_states = model.state_dict()
    save_checkpoint(epoch, save_states, optimizer.state_dict(), logger.logdir)    
    if (epoch % P.save_step == 0):
        torch.cuda.empty_cache()
        from evals.ood_pre import eval_ood_detection
        P.load_path = logger.logdir + '/last.model'
        import subprocess

        arguments_to_pass = [
            "--image_size", str(P.image_size[0]),
            "--dataset", str(P.dataset),
            "--model", str(P.model),
            "--print_score",
            "--resize_fix",
            "--ood_samples", "10",
            "--resize_factor", str(0.54),
            "--one_class_idx" , str(P.one_class_idx),
            "--load_path", str(P.load_path),
            "--normal_label", str(P.normal_label),
            "--noise_scale",str(0.0),
            "--noist_probability", str(0.0),
            '--activation_function', str(P.activation_function)
        ]

        result = subprocess.run(["python", "eval.py"] + arguments_to_pass, capture_output=True, text=True)

        # Check the result
        if result.returncode == 0:
            logger.log("Script executed successfully.")
            logger.log("Output:")
            logger.log(result.stdout)
        else:
            logger.log("Script execution failed.")
            logger.log("Error:")
            logger.log(result.stderr)
        
epoch += 1
if P.multi_gpu:
    save_states = model.module.state_dict()
else:
    save_states = model.state_dict()
save_checkpoint(epoch, save_states, optimizer.state_dict(), logger.logdir)
save_linear_checkpoint(linear_optim.state_dict(), logger.logdir)

