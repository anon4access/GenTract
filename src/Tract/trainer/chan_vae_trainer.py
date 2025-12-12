import os
import warnings

import wandb
import torch
from torch.amp import autocast, GradScaler
from torch.nn import L1Loss
from tqdm import tqdm

from generative.losses import PerceptualLoss, PatchAdversarialLoss
from Tract.utils import AverageLoss
from Tract.utils.visualizer import log_reconstruction
from Tract.utils.torch_utils import KLDivergenceLoss


def train_loop(
    autoencoder,
    discriminator,
    train_loader,
    valid_loader,
    optimizer_d,
    optimizer_g,
    scheduler_d,
    scheduler_g,
    scaler_g,
    scaler_d,
    device,
    cfg,
    results_dir,
    params,
    sh_coeff,
    start_epoch,
    total_counter,
    best_val_loss
):
    # GradScalers, best_val_loss, and total_counter are now passed in.


    print('Running with best validation loss: ', best_val_loss)
    
    
    if params == 'brlp':
        adv_weight = 0.025       
        perceptual_weight = 0.001
        kl_weight = 1e-7
    else:
        adv_weight = 0.1      
        perceptual_weight = 0.1
        kl_weight = 1e-6

    device_str = str(device)
    l1_loss_fn  = L1Loss()
    kl_loss_fn  = KLDivergenceLoss()
    adv_loss_fn = PatchAdversarialLoss(criterion="least_squares")
    val_interval = cfg['training']['val_interval']

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perc_loss_fn = PerceptualLoss(spatial_dims=3, 
                                      network_type="squeeze", 
                                      is_fake_3d=True, 
                                      fake_3d_ratio=0.2).to(device)

    avgloss = AverageLoss()
    
    # --- MODIFIED: Start loop from the correct epoch ---
    
    print('Starting at epoch: ', start_epoch)
    for epoch in range(start_epoch, cfg['training']['n_epochs']):
        autoencoder.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        progress_bar.set_description(f'Epoch {epoch}')

        for step, batch in progress_bar:

            optimizer_g.zero_grad(set_to_none=True)
            optimizer_d.zero_grad(set_to_none=True)
            
            with autocast(device_str, enabled=True):
                images = batch["image"].to(device)
                print("TEST IMAGE SHAPE ", images.shape)
                reconstruction, z_mu, z_sigma = autoencoder(images)
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                
                rec_loss = l1_loss_fn(reconstruction.float(), images.float())
                kld_loss = kl_weight * kl_loss_fn(z_mu, z_sigma)
                per_loss = perceptual_weight * perc_loss_fn(reconstruction.float(), images.float())
                gen_loss = adv_weight * adv_loss_fn(logits_fake, target_is_real=True, for_discriminator=False)
                
                loss_g = rec_loss + kld_loss + per_loss + gen_loss
            
            scaler_g.scale(loss_g).backward()
            scaler_g.step(optimizer_g)
            scaler_g.update()
            
            with autocast(device_str, enabled=True):
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                d_loss_fake = adv_loss_fn(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                d_loss_real = adv_loss_fn(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (d_loss_fake + d_loss_real) * 0.5
                loss_d = adv_weight * discriminator_loss
            
            scaler_d.scale(loss_d).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()
            
            avgloss.put('Generator/reconstruction_loss', rec_loss.item())
            avgloss.put('Generator/perceptual_loss', per_loss.item())
            avgloss.put('Generator/adversarial_loss', gen_loss.item())
            avgloss.put('Generator/kl_regularization', kld_loss.item())
            avgloss.put('Discriminator/adversarial_loss', loss_d.item())
            
            if total_counter % 10 == 0:
                avgloss.to_wandb(total_counter)
            
            total_counter += 1
        
        scheduler_d.step()
        scheduler_g.step()
        
        if (epoch+1) % val_interval == 0:

            print('Running validation.')
            autoencoder.eval()
            valid_recon_losses = []
            
            with torch.no_grad():
                for batch in tqdm(valid_loader, total=len(valid_loader)):
                    with autocast(device_str, enabled=True):
                        images = batch["image"].to(device)
                        reconstruction = autoencoder.reconstruct(images)
                        rec_loss = l1_loss_fn(reconstruction.float(), images.float())
                        valid_recon_losses.append(rec_loss.item())                    

            valid_epoch_recon_loss = sum(valid_recon_losses) / len(valid_recon_losses)
            print(f"Validation Reconstruction Loss: {valid_epoch_recon_loss:.6f}")
            
            wandb.log({'Valid/ReconLoss': valid_epoch_recon_loss}, step=total_counter)
                    
            if valid_epoch_recon_loss < best_val_loss:
                best_val_loss = valid_epoch_recon_loss
                
                print("New best validation loss. Saving training state.")
                
                # --- NEW: Save the comprehensive training checkpoint ---
                training_state = {
                    'epoch': epoch,
                    'total_counter': total_counter,
                    'autoencoder_state_dict': autoencoder.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'optimizer_g_state_dict': optimizer_g.state_dict(),
                    'optimizer_d_state_dict': optimizer_d.state_dict(),
                    'scheduler_g_state_dict': scheduler_g.state_dict(),
                    'scheduler_d_state_dict': scheduler_d.state_dict(),
                    'scaler_g_state_dict': scaler_g.state_dict(),
                    'scaler_d_state_dict': scaler_d.state_dict(),
                    'best_val_loss': best_val_loss,
                }
                torch.save(training_state, os.path.join(results_dir, "training_checkpoint.pth"))
                
                # Keep saving these for easy inference access
                torch.save(autoencoder.state_dict(), os.path.join(results_dir, "best_autoencoder.pth"))
                torch.save(discriminator.state_dict(), os.path.join(results_dir, "best_discriminator.pth"))

                try:
                    log_reconstruction(total_counter, images[0].detach().cpu(), reconstruction[0].detach().cpu(), coeff=sh_coeff, save_dir=results_dir)
                except Exception as e:
                    print(f"There was an error during visualization: {e}. Resuming training...")
                    continue


def test_loop(
    autoencoder,
    test_loader,
    device,
    results_dir
):
    device_str = str(device)
    print('Running test.')
    autoencoder.eval()
    test_recon_losses = []
    l1_loss_fn  = L1Loss()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader)):
            with autocast(device_str, enabled=True):
                images = batch["image"].to(device)
                reconstruction = autoencoder.reconstruct(images)
                rec_loss = l1_loss_fn(reconstruction.float(), images.float())
                test_recon_losses.append(rec_loss.item())                    

    test_loss = sum(test_recon_losses) / len(test_recon_losses)
    print(f"Test Reconstruction Loss: {test_loss:.6f}")
    wandb.log({'Test/ReconLoss': test_loss})