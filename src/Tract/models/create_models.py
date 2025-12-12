import torch
from generative.networks.nets import AutoencoderKL
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
from monai.networks.nets import PatchDiscriminator
from .cond_diff import DiffusionTransformer



def create_maisi_vae(device, cfg):
    """
    Instantiate the MAISI VAE-GAN using the 
    original configurations provided in MONAI's demo.
    """
    vae = AutoencoderKlMaisi(
        spatial_dims=3,
        in_channels=1, 
        out_channels=1,
        latent_channels=4,
        num_channels=[64, 128, 256],
        num_res_blocks=[2, 2, 2],
        norm_num_groups=32,
        norm_eps=1e-06,
        attention_levels=[False, False, False],
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
        use_checkpointing=False,
        use_convtranspose=False,
        norm_float16=True,
        num_splits=2,
        dim_split=1
    ).to(device)
    
    # Load the pretrained checkpoints
    vae.load_state_dict(torch.load(cfg["vae"]["pretrained"], map_location=device, weights_only=True))
    return vae


def create_patch_discriminator(device):
    """
    """
    discriminator = PatchDiscriminator(
        spatial_dims=3,
        num_layers_d=3,
        channels=32,
        in_channels=1,
        out_channels=1,
        norm="INSTANCE",
    ).to(device)
    return discriminator


def create_diffusion(device, cfg):
    model = DiffusionTransformer(**cfg['model'])
    return model.to(device) 