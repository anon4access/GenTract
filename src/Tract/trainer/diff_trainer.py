import os
import math
import warnings
import wandb
import torch
from torch.amp import GradScaler, autocast
from torch.nn import MSELoss
from tqdm import tqdm
from Tract.utils import AverageLoss
from Tract.utils.visualizer import log_generation
from Tract.utils.streamlines import apply_affine_transform_torch, generate_rotation_matrix_torch
import nibabel as nib
import numpy as np

def train_loop(
    model,
    inferer,
    diffusion_scheduler,
    optimizer,
    train_loader,
    valid_loader,
    lr_scheduler,
    device,
    cfg,
    results_dir
    ):

    # Pre-compute all rotation matrices on the correct device
    print("Pre-computing rotation matrices...")
    rotation_matrices = {}
    axes = [0, 1, 2]
    angles = [-45.0, -30.0, -15.0, 15.0, 30.0, 45.0]
    for axis in axes:
        for angle in angles:
            rotation_matrices[(axis, angle)] = generate_rotation_matrix_torch(axis, angle, device)
    rotation_matrices[(0, 0.0)] = torch.eye(4, dtype=torch.float32, device=device)
    print(f"Generated {len(rotation_matrices)} matrices.")

    scaler = GradScaler()
    conditioning = cfg["model"]["cond"]
    best_val_loss = float('inf')
    device_str = str(device)
    mse_loss  = MSELoss()
    avgloss = AverageLoss()
    total_counter = 0
    num_timesteps = diffusion_scheduler.num_train_timesteps

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

            # Apply rotation on-the-fly for training
            rot_axis = batch["rot"].item()
            deg = batch["deg"].item()

            if deg != 0.0:
                transform_matrix = rotation_matrices.get((rot_axis, deg))
                if transform_matrix is not None:
                    with torch.no_grad(): # Don't track gradients for augmentation
                        x = apply_affine_transform_torch(x, transform_matrix)
                else:
                    warnings.warn(f"Warning: No rotation matrix found for axis={rot_axis}, deg={deg}. Skipping rotation.")

            timesteps = torch.randint(0, num_timesteps, (x.shape[0], ), device=device).long()

            with autocast(device_str, enabled=True):
                noise = torch.randn_like(x).to(device)
                noise_pred = inferer(
                    inputs=x,
                    diffusion_model=model,
                    noise=noise,
                    timesteps=timesteps,
                    condition=context
                    )
                loss = mse_loss(noise_pred.float(), noise.float())

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

        # --- VALIDATION LOOP ---
        print('Running validation.')
        model.eval()
        valid_losses = []

        with torch.no_grad():
            for batch in tqdm(valid_loader, total=len(valid_loader)):
                with autocast(device_str, enabled=True):
                    x = batch["tracts"].to(device).squeeze(0)

                    if conditioning:
                        context = batch["latent"].to(device)

                    # --- ADDED: Apply rotation on-the-fly for validation ---
                    rot_axis = batch["rot"].item()
                    deg = batch["deg"].item()

                    if deg != 0.0:
                        transform_matrix = rotation_matrices.get((rot_axis, deg))
                        if transform_matrix is not None:
                            x = apply_affine_transform_torch(x, transform_matrix)
                        else:
                            warnings.warn(f"Warning: No validation rotation matrix found for axis={rot_axis}, deg={deg}.")
                    # --- END ADDED ---

                    timesteps = torch.randint(0, num_timesteps, (x.shape[0],), device=device).long()
                    noise = torch.randn_like(x).to(device)
                    noise_pred = inferer(
                        inputs=x,
                        diffusion_model=model,
                        noise=noise,
                        timesteps=timesteps,
                        condition=context
                        )
                    loss = mse_loss(noise_pred.float(), noise.float())
                    valid_losses.append(loss.item())

        valid_loss = sum(valid_losses) / len(valid_losses)
        print(f"Validation Reconstruction Loss: {valid_loss:.6f}")
        wandb.log({'Valid/mse_loss': valid_loss, "epoch": epoch+1})

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            print("New best validation loss. Saving training state.")
            torch.save(
                model.state_dict(),
                os.path.join(results_dir, "best_model.pth")
            )
            try:
                xshape = list(x.shape)
                log_generation(epoch, model, xshape, context, cfg, device,
                            save_dir=results_dir)
            except Exception as e:
                print("There was an error: ", e)
                print("Resuming training ...")
                continue