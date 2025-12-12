
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import argparse
import shutil

import torch
import torchinfo
import wandb
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR


from Tract.data import prepare_diff_data
from Tract.models.create_models import create_diffusion
from Tract.utils import setup_torch, load_config
from Tract.trainer.fm_trainer import train_loop



def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--temp_dir", type=str, required=True, help="Scratch dir")

    args = parser.parse_args()
    
    #===================================================
    # Settings.
    #===================================================
    
    cfg          = load_config(args.config)    
    device       = setup_torch()
    num_coeffs   = cfg["model"]["M"]
    num_tl       = cfg["model"]["diff_transformer_layers"]
    model_dim    = cfg["model"]["model_dim"]
    num_samples  = cfg["data"]["num_samples"]
    conditioning = cfg["model"]["cond"]

    name         = f'fm_{model_dim}_{num_tl}'

    print(f"Using device: {device}")    
    print(f'Running with run name: {name}')

    #===================================================
    # Initialising wandb
    #===================================================

    wandb.init(
         project=cfg["wandb"]["project"], 
         config=cfg, 
         name=name
     )    

    #===================================================
    # Preparing datasets and dataloaders
    #===================================================
    print('Preparing datasets and dataloaders, with m: ', num_coeffs)

        
    trainset, validset, testset = prepare_diff_data(
        csv_file=cfg["data"]["csv_file"],
        temp_dir=args.temp_dir,
        cache=cfg["data"]["cache"],
        num_coeffs=num_coeffs,
        num_samples=num_samples,
        cond=conditioning,
        streamline_csv=cfg["data"]["streamline_csv"]
                )
    
    batch_size = cfg['training']['batch_size'] 
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,  num_workers=8, pin_memory=False)
    valid_loader = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False)
    test_loader  = DataLoader(testset,  batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False)

    
    print("DataLoaders initialized.")
    
    #===================================================
    # Initialising the models
    #===================================================
    print('Initialising the models with conditioning: ', conditioning)
    
    model = create_diffusion(device, cfg) # ADD CREATE DIFFUSION MODEL
    
    model.train()

    print("\n--- torchinfo summary ---")

    torchinfo.summary(model)

    #===================================================
    # Creating the result directory
    #===================================================
    print("Creating the result directory.")

    base_dir    = cfg["paths"]["base_dir"]
    results_dir = os.path.join(base_dir, "results", name)
    os.makedirs(results_dir, exist_ok=True)
    
    config_copy_path = os.path.join(results_dir, os.path.basename(args.config))
    shutil.copy(args.config, config_copy_path)
    
    print(f"Results directory created: {results_dir}")
    print(f"Configuration file copied to: {config_copy_path}")


    check_dir = os.path.join(base_dir, "results", f'fm_{model_dim}_{num_tl}')

    #===================================================
    # Load model checkpoint
    #===================================================
    checkpoint_path = os.path.join(check_dir,'')
    print('Loading model from checkpoint', checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True), strict=False)

    print('Model loaded')


    #===================================================
    # Print summary of the first batch
    #===================================================

    batch = next(iter(train_loader))
    
    if conditioning:

        print("Image shape: ", batch["latent"].shape)
    
    print("Tracts shape", batch["tracts"].shape)

    #======================================================
    # Initialise the optimizers, schedulers, and gradient accumulators
    #======================================================

    # initialise the optimizers 
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['lr'])


    scheduler = CosineAnnealingLR(optimizer, 
                                  T_max=cfg["training"]["n_epochs"], 
                                  eta_min=cfg["training"]["lr"] / 4)


    #======================================================
    # Run the training loop
    #======================================================

    train_loop(
        model=model, 
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr_scheduler=scheduler,
        device=device,
        cfg=cfg,
        results_dir=results_dir
        )

    

    wandb.finish()



if __name__ == "__main__":
    main()
