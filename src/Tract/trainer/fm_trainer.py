import os
import math # <-- Added import
import warnings # <-- Added import

import wandb
import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from flow_matching.path import CondOTProbPath

from Tract.utils import AverageLoss
from Tract.utils.visualizer import log_generation

from Tract.utils.streamlines import apply_affine_transform_torch, generate_rotation_matrix_torch




def skewed_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    """
    Timestep sampling function proposed in the EDM paper (https://arxiv.org/abs/2206.00364)
    """
    P_mean = -1.2
    P_std = 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    time = 1 / (1 + sigma)
    time = torch.clip(time, min=0.0001, max=1.0)
    return time

def compute_fm_loss(model, fm_path, x, context, device):
    """
    Compute the conditional FM loss.
    """
    # we use the timestep embedding sampling from the EDM paper.
    timesteps = skewed_timestep_sample(x.shape[0], device=device)

    # sample noise (prior distribution)
    noise = torch.randn_like(x).to(device)
    # xt = (1-t)x0 + tx1 and the conditional velocity field
    # u_t = x1 - x0.
    path_sample = fm_path.sample(t=timesteps, x_0=noise, x_1=x)
    x_t = path_sample.x_t
    true_u_t = path_sample.dx_t

    with autocast(device, enabled=True):
        # we estimate the conditional velocity field using our model.
        pred_u_t = model(x=x_t, timesteps=timesteps, context=context)

        # and then calculate the conditional FM loss.
        return torch.pow(pred_u_t - true_u_t, 2).mean()


def train_loop(
    model,
    optimizer,
    train_loader,
    valid_loader,
    lr_scheduler,
    device,
    cfg,
    results_dir
    ):

    # --- NEW: Pre-compute all rotation matrices on the correct device ---
    print("Pre-computing rotation matrices...")
    rotation_matrices = {}
    axes = [0, 1, 2]
    angles = [-45.0, -30.0, -15.0, 15.0, 30.0, 45.0]
    for axis in axes:
        for angle in angles:
            rotation_matrices[(axis, angle)] = generate_rotation_matrix_torch(axis, angle, device)
    # Add identity matrix for non-rotated case (deg=0)
    rotation_matrices[(0, 0.0)] = torch.eye(4, dtype=torch.float32, device=device)
    print(f"Generated {len(rotation_matrices)} matrices.")
    # --- END NEW ---

    scaler = GradScaler()
    conditioning = cfg["model"]["cond"]
    scaler = GradScaler()
    conditioning = cfg["model"]["cond"]
    M = cfg["model"]["M"]
    best_val_loss = float('inf')
    device_str = str(device)
    path = CondOTProbPath()


    avgloss, val_avgloss = AverageLoss(), AverageLoss()
    total_counter = 0

    if not conditioning:
        context = None

    for epoch in range(cfg['training']['n_epochs']):
        model.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        progress_bar.set_description(f'Epoch {epoch}')

        train_loss = 0.0
        train_batches = 0

        for step, batch in progress_bar:

            optimizer.zero_grad()
            train_batches += 1

            x = batch["tracts"].to(device).squeeze(0)

            if conditioning:
                context = batch["latent"].to(device)

            rot_axis = batch["rot"].item()
            deg = batch["deg"].item()

            if deg != 0.0:
                transform_matrix = rotation_matrices.get((rot_axis, deg))
                if transform_matrix is not None:
                    with torch.no_grad(): # Don't track gradients for augmentation
                        x = apply_affine_transform_torch(x, transform_matrix)
                else:
                    warnings.warn(f"Warning: No training rotation matrix found for axis={rot_axis}, deg={deg}.")

            loss = compute_fm_loss(
                model=model, 
                fm_path=path,
                x=x,
                context=context,
                device=device_str
            )

            train_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            avgloss.put('Diffusion/mse_loss', loss.item())

            
            if total_counter % 10 == 0:
                avgloss.to_wandb(total_counter)
            
            total_counter += 1

        avg_train_loss = train_loss / train_batches
        print(f"Avg Training Loss: {avg_train_loss:.6f}")

        lr_scheduler.step()

        print('Running validation.')
        model.eval()
        valid_losses = []

        with torch.no_grad():
            for batch in tqdm(valid_loader, total=len(valid_loader)):
                with autocast(device_str, enabled=True):
                    x = batch["tracts"].to(device).squeeze(0)
                    if conditioning:
                        context = batch["latent"].to(device)

                    # --- NEW: Apply rotation on-the-fly for validation ---
                    rot_axis = batch["rot"].item()
                    deg = batch["deg"].item()

                    if deg != 0.0:
                        transform_matrix = rotation_matrices.get((rot_axis, deg))
                        if transform_matrix is not None:
                            x = apply_affine_transform_torch(x, transform_matrix)
                        else:
                            warnings.warn(f"Warning: No validation rotation matrix found for axis={rot_axis}, deg={deg}.")
                    # --- END NEW ---

                    loss = compute_fm_loss(
                        model=model, 
                        fm_path=path,
                        x=x,
                        context=context,
                        device=device_str
                    )
                    valid_losses.append(loss.item())

        valid_loss = sum(valid_losses) / len(valid_losses)
        print(f"Validation Reconstruction Loss: {valid_loss:.6f}")

        # Log the loss to wandb
        wandb.log({'Valid/mse_loss': valid_loss,  "epoch": epoch+1 })

      
        # Save the best model
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            
            print("New best validation loss. Saving training state.")

            
            torch.save(
                model.state_dict(), 
                os.path.join(results_dir, "best_model.pth'")
            )
            
            try:
                xshape = list(x.shape)
                log_generation(epoch, model, xshape, context, cfg, device, 
                               save_dir=results_dir,
                               gen_framework='flow_matching')
                
            except Exception as e:
                print("There was an error: ", e)
                print("Resuming training ...")
                continue