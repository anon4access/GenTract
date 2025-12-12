import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import argparse
import shutil

import torch
import wandb
from torch.utils.data import DataLoader
from torch.amp import GradScaler # Moved here from trainer
from torch.optim.lr_scheduler import CosineAnnealingLR

from Tract.data import prepare_coeff_data
from Tract.models.create_models import create_maisi_vae, create_patch_discriminator
from Tract.utils import setup_torch, load_config
# Make sure to import from your trainer file, e.g., 'trainer.py'
from Tract.trainer.chan_vae_trainer import train_loop, test_loop


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--coeff", type=int, required=True, help="sh coeff")
    parser.add_argument("--params", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    # --- NEW: Argument to trigger resume ---
    parser.add_argument("--resume", action="store_true", help="Resume training from a checkpoint")
    args = parser.parse_args()
    
    #===================================================
    # Settings.
    #===================================================
    
    cfg         = load_config(args.config)    
    device      = setup_torch()
    res         = cfg["data"]["resolution"]
    coeff       = args.coeff
    params      = args.params
    model_type  = cfg["training"]["model"]
    name        = f'vae_{model_type}-coeff_{coeff}'
    data_dir    = args.data_dir
    print(f"Using device: {device}")    
    print(f'Running with run name: {name}')

    #===================================================
    # Initialising wandb
    #===================================================

    wandb.init(
         project=cfg["wandb"]["project"], 
         config=cfg, 
         name=name,
         resume="allow" # Allows wandb to resume if the run exists
     )    

    #===================================================
    # Preparing datasets and dataloaders
    #===================================================
    print('Reading data from directory: ', data_dir)
    print('Preparing datasets and dataloaders.')
        
    trainset, validset, testset = prepare_coeff_data(
        model_type=model_type,
        data_dir=data_dir,
        num_unet_layers=4,
        resolution=res,
        cache=cfg["data"]["cache"],
        coeff=coeff,
        augment_train=False,
        bbox_csv=cfg["data"]["bbox_csv"],
        stats_csv=cfg["data"]["stats_csv"]
    )
    
    batch_size = cfg['training']['batch_size'] 
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,  num_workers=8, pin_memory=False)
    valid_loader = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False)
    test_loader  = DataLoader(testset,  batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False)
    
    print("DataLoaders initialized.")
    
    #===================================================
    # Initialising the models
    #===================================================
    print('Initialising the models.')
    
    autoencoder = create_maisi_vae(device, cfg)
    autoencoder.train()
    
    discriminator = create_patch_discriminator(device)
    discriminator.train()    

    print('Models initialised')
    #===================================================
    # Creating the result directory
    #===================================================
    print("Creating the result directory.")

    base_dir    = cfg["paths"]["base_dir"]
    results_dir = os.path.join(base_dir, "results", name)
    os.makedirs(results_dir, exist_ok=True)
    
    if not args.resume: # Only copy config on a fresh run
        config_copy_path = os.path.join(results_dir, os.path.basename(args.config))
        shutil.copy(args.config, config_copy_path)
        print(f"Configuration file copied to: {config_copy_path}")
    
    print(f"Results directory: {results_dir}")

    #======================================================
    # Initialise the optimizers and schedulers
    #======================================================

    optimizer_g = torch.optim.Adam(autoencoder.parameters(),   lr=cfg['training']['lr'])
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=cfg['training']['lr'])

    scheduler_g = CosineAnnealingLR(optimizer_g, T_max=cfg["training"]["n_epochs"], eta_min=cfg["training"]["lr"] / 2)
    scheduler_d = CosineAnnealingLR(optimizer_d, T_max=cfg["training"]["n_epochs"], eta_min=cfg["training"]["lr"] / 2)

    #======================================================
    # --- NEW: Prepare for Resuming Training ---
    #======================================================
    
    scaler_g = GradScaler()
    scaler_d = GradScaler()

    start_epoch = 0
    total_counter = 0
    best_val_loss = float('inf')
    
    checkpoint_path = os.path.join(results_dir, "training_checkpoint.pth")

    if args.resume and os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
        scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])
        scaler_g.load_state_dict(checkpoint['scaler_g_state_dict'])
        scaler_d.load_state_dict(checkpoint['scaler_d_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        total_counter = checkpoint['total_counter']
        best_val_loss = checkpoint['best_val_loss']
        
        print(f"Resumed from epoch {checkpoint['epoch']}. Starting next epoch at {start_epoch}.")
    else:
        print("Starting training from scratch.")

    #======================================================
    # Run the training loop
    #======================================================

    train_loop(
        autoencoder=autoencoder, 
        discriminator=discriminator,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer_d=optimizer_d,
        optimizer_g=optimizer_g,
        scheduler_d=scheduler_d,
        scheduler_g=scheduler_g,
        scaler_g=scaler_g,
        scaler_d=scaler_d,
        device=device,
        cfg=cfg,
        results_dir=results_dir,
        params=params,
        sh_coeff=coeff,
        start_epoch=start_epoch,
        total_counter=total_counter,
        best_val_loss=best_val_loss
    )

    #======================================================
    # Run the test loop
    #======================================================

    print("Loading best model for testing.")
    best_autoencoder_path = os.path.join(results_dir, "best_autoencoder.pth")
    autoencoder.load_state_dict(torch.load(best_autoencoder_path, map_location=device))

    test_loop(
        autoencoder=autoencoder,
        test_loader=test_loader,
        device=device,
        results_dir=results_dir
    )

    wandb.finish()


if __name__ == "__main__":
    main()