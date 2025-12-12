# my_diffusion/utils/visualizer.py
import os
import wandb
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from .sampling import sample_using_diffusion, sampling_using_fm


def save_nifti_image(data, save_path):
    """Save a numpy array as a NIfTI image."""
    nii_img = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(nii_img, save_path)
    print(f"Saved NIfTI image to {save_path}")

def plot_reconstruction(original, generated, save_dir, filename='gen_pet.png'):
    """
    Plot side-by-side slices of original and generated images.
    """
    depth = original.shape[-1]
    slice_indices = [int(np.round(depth / 4)),
                     int(np.round(depth / 2)),
                     int(np.round(depth * 3 / 4))]

    orig_min, orig_max = original.min(), original.max()
    gen_min, gen_max = generated.min(), generated.max()

    fig, axes = plt.subplots(3, 2, figsize=(8, 12))
    for i, slice_idx in enumerate(slice_indices):
        axes[i, 0].imshow(original[:, :, slice_idx], cmap='gray', vmin=orig_min, vmax=orig_max)
        axes[i, 0].set_title(f'Original Slice {slice_idx}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(generated[:, :, slice_idx], cmap='gray', vmin=gen_min, vmax=gen_max)
        axes[i, 1].set_title(f'Reconstructed Slice {slice_idx}')
        axes[i, 1].axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Reconstruction plot saved to {save_path}")

    
    
def log_reconstruction(step, image, recon, coeff, save_dir):
    """
    Display reconstruction in TensorBoard during AE training.
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(7, 5))
    for _ax in ax.flatten(): _ax.set_axis_off()

    if len(image.shape) == 4: image = image.squeeze(0) 
    if len(recon.shape) == 4: recon = recon.squeeze(0)

    ax[0, 0].set_title(f'original image - c_{coeff}', color='magenta')
    ax[0, 0].imshow(image[image.shape[0] // 2, :, :], cmap='gray')
    ax[0, 1].imshow(image[:, image.shape[1] // 2, :], cmap='gray')
    ax[0, 2].imshow(image[:, :, image.shape[2] // 2], cmap='gray')

    ax[1, 0].set_title(f'reconstructed image - c_{coeff} ', color='cyan')
    ax[1, 0].imshow(recon[recon.shape[0] // 2, :, :], cmap='gray')
    ax[1, 1].imshow(recon[:, recon.shape[1] // 2, :], cmap='gray')
    ax[1, 2].imshow(recon[:, :, recon.shape[2] // 2], cmap='gray')

    fig.tight_layout()
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'best_recon.png')
    plt.savefig(save_path)
    print('Reconstructions logged to: ', save_path)
    wandb.log({"plot": wandb.Image(fig)}, step=step)
    plt.close()


def log_generation(
        epoch, 
        diffusion, 
        xshape, 
        context, 
        cfg, 
        device, 
        save_dir, 
        gen_framework='diffusion', 
        sigma_data=None,
        steps=None,
        wandb_name='plot'
    ):
    """
    Visualize the generation on tensorboard
    """
    diffusion.eval()
    
    if gen_framework == 'diffusion':
        tracts = sample_using_diffusion(
            xshape=xshape,
            context=context,
            diffusion=diffusion, 
            device=device,
            num_training_steps=cfg["training"]["timesteps"],
            num_inference_steps=cfg["diffusion"]["num_inference_steps"],
            beta_start=cfg["diffusion"]["beta_start"],
            beta_end=cfg["diffusion"]["beta_end"]
        )
    elif gen_framework == 'flow_matching':
        tracts = sampling_using_fm(
            xshape=xshape,
            context=context,
            model=diffusion,
            device=device,
        )

    else:
        raise Exception(f'Method {gen_framework} not implemented.')

    tracts = tracts.detach().cpu().numpy()
            

    fig = plt.figure(figsize=(18, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_title('Generated streamlines')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    for tract in tracts:

        x, y, z = tract[:, 0], tract[:, 1], tract[:, 2]

        ax.plot(x, y, z, color='red', alpha=0.6)
    ax.view_init(elev=20, azim=120)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(save_dir, f'best_{wandb_name}.png')
    plt.savefig(save_path)
    print('Generations logged to: ', save_path)
    wandb.log({f'Plots/{wandb_name}': wandb.Image(fig)})
    plt.close()