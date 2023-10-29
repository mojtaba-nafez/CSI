from utils.utils import Logger, save_checkpoint, save_linear_checkpoint
import os
from common.train import *
from evals import test_classifier
from common.swa import initialize_swa_model, get_swa_scheduler, update_swa_model, update_batch_norm

if 'sup' in P.mode:
    from training.sup import setup
else:
    from training.unsup import setup
    
train, fname = setup(P.mode, P)

logger = Logger(fname, ask=not resume, local_rank=P.local_rank)
logger.log(P)
logger.log(model)

if P.multi_gpu:
    linear = model.module.linear
else:
    linear = model.linear
linear_optim = torch.optim.Adam(linear.parameters(), lr=1e-3, betas=(.9, .999), weight_decay=P.weight_decay)

# Initialize the SWA model
swa_model = initialize_swa_model(model.cuda()).cuda()

# Update your LR scheduler for SWA
swa_scheduler = get_swa_scheduler(optimizer, P.swa_lr)

epoch = 0
# Run experiments
for epoch in range(start_epoch, P.epochs + 1 + P.swa_epochs):
    logger.log_dirname(f"Epoch {epoch}")
    model.train()

    if P.multi_gpu:
        train_sampler.set_epoch(epoch)

    kwargs = {
        'linear': linear,
        'linear_optim': linear_optim,
        'simclr_aug': simclr_aug
    }

    if epoch > P.unfreeze_pretrain_model_epoch:
        for param in model.parameters():
            param.requires_grad = True

    train(P, epoch, model.cuda(), criterion, optimizer, scheduler_warmup, train_loader, train_exposure_loader=train_exposure_loader, logger=logger,
          swa_model=swa_model.cuda(), swa_scheduler=swa_scheduler, swa_start=P.start_swa_epoch, swa_update_frequency=P.swa_update_frequency, **kwargs)
        
    if (epoch % P.save_step == 0):
        # Update batch normalization layers for SWA model
        os.makedirs(os.path.join(logger.logdir, 'swa_model'), exist_ok=True)
        update_batch_norm(train_loader, swa_model)
        save_states = swa_model.module.state_dict()
        save_checkpoint(epoch, save_states, optimizer.state_dict(), os.path.join(logger.logdir, 'swa_model')) 
    
        torch.cuda.empty_cache()
        from evals.ood_pre import eval_ood_detection
        P.load_path = os.path.join(logger.logdir, 'swa_model', 'last.model')
        import subprocess

        arguments_to_pass = [
            "--image_size", str(P.image_size[0]),
            "--mode", "ood_pre",
            "--dataset", str(P.dataset),
            "--model", str(P.model),
            "--ood_score", "CSI",
            "--shift_trans_type", "rotation",
            "--print_score",
            "--ood_samples", "10",
            "--resize_factor", str(0.54),
            "--resize_fix", 
            "--one_class_idx" , str(P.one_class_idx),
            "--load_path", str(P.load_path),
            "--normal_labels", str(P.normal_labels),
            "--noise_scale",str(0.0),
            "--noist_probability", str(0.0),
            '--activation_function', str(P.activation_function)
        ]

        result = subprocess.run(["python", "eval.py"] + arguments_to_pass, capture_output=True, text=True)

        if result.returncode == 0:
            logger.log("Script executed successfully.")
            logger.log("Output:")
            logger.log(result.stdout)
        else:
            logger.log("Script execution failed.")
            logger.log("Error:")
            logger.log(result.stderr)

# Update batch normalization layers for SWA model
update_batch_norm(train_loader, swa_model)
     
epoch += 1
if P.multi_gpu:
    save_states = model.module.state_dict()
else:
    save_states = model.state_dict()
    
# Save SWA Model
swa_save_states = swa_model.module.state_dict()
save_checkpoint(epoch, swa_save_states, optimizer.state_dict(), os.path.join(logger.logdir, 'swa_model'))
