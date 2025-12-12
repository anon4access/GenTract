import math
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from monai import transforms
from monai.data import MetaTensor
from monai.networks.schedulers.ddim import DDIMScheduler
from tqdm import tqdm
from flow_matching.solver import ODESolver

from Tract.models.cond_diff import VelocityModelWrapper
# from Tract.utils.cm_utils import f_theta


@torch.no_grad()
def sample_using_diffusion(
    xshape: tuple,
    context: torch.tensor,
    diffusion: nn.Module, 
    device: str, 
    num_training_steps: int = 1000,
    num_inference_steps: int = 50,
    schedule: str = 'scaled_linear_beta',
    beta_start: float = 0.0015, 
    beta_end: float = 0.0205, 
    verbose: bool = True,
) -> torch.Tensor: 
    """
    Sampling random tractss that follow the dmri conditioning in `context`.

    Args:
        xshape: shape of the tracts used during training including the channel dimension
        diffusion (nn.Module): the UNet 
        device (str): the device ('cuda' or 'cpu')
        num_training_steps (int, optional): T parameter. Defaults to 1000.
        num_inference_steps (int, optional): reduced T for DDIM sampling. Defaults to 50.
        schedule (str, optional): noise schedule. Defaults to 'scaled_linear_beta'.
        beta_start (float, optional): noise starting level. Defaults to 0.0015.
        beta_end (float, optional): noise ending level. Defaults to 0.0205.
        verbose (bool, optional): print progression bar. Defaults to True.
    Returns:
        torch.Tensor: the inferred tracts
    """
    scheduler = DDIMScheduler(num_train_timesteps=num_training_steps,
                              schedule=schedule,
                              beta_start=beta_start,
                              beta_end=beta_end,
                              clip_sample=False)

    scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    # drawing a random z_T ~ N(0,I)
    x = torch.randn(xshape).to(device)
    
    progress_bar = tqdm(scheduler.timesteps) if verbose else scheduler.timesteps
    for t in progress_bar:
        with torch.no_grad():
            with autocast(enabled=True):

                timestep = torch.tensor([t]).to(device)
                
                # predict the noise
                noise_pred = diffusion(x=x.float(), context=context, timesteps=timestep)

                # the scheduler applies the formula to get the 
                # denoised step z_{t-1} from z_t and the predicted noise
                x, _ = scheduler.step(noise_pred, t, x)
    
    
    return x


@torch.no_grad()
def sampling_using_fm(
    xshape,
    model,
    device,
    atol=1e-5,
    rtol=1e-5,
    ode_method='midpoint',
    nfe=512,
    context=None

):
    """
    """
    # wrap the existing model to allow working with the ODE solver.
    wrapped_model = VelocityModelWrapper(model)
    
    # instantiate a standard ODESolver.
    solver = ODESolver(velocity_model=wrapped_model)
    
    # sample from the prior
    x_0 = torch.randn(xshape, dtype=torch.float32, device=device)

    # use the time discretization from the EDM paper
    #time_grid = get_time_discretization(nfe)

    time_grid = torch.linspace(0.0, 1.0, nfe, device=device)

    step_size = 1/nfe
    time_grid = get_time_discretization(nfe)

    return solver.sample(
        time_grid=time_grid,
        step_size=step_size,
        x_init=x_0,
        method=ode_method,
        return_intermediates=False,
        atol=atol,
        rtol=rtol,
        context=context
                        )

    
def get_time_discretization(nfes: int, rho=7):
    """
    Time discretization from the EDM paper.
    """
    step_indices = torch.arange(nfes, dtype=torch.float64)
    sigma_min = 0.002
    sigma_max = 80.0
    sigma_vec = (
        sigma_max ** (1 / rho)
        + step_indices / (nfes - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    sigma_vec = torch.cat([sigma_vec, torch.zeros_like(sigma_vec[:1])])
    time_vec = (sigma_vec / (1 + sigma_vec)).squeeze()
    t_samples = 1.0 - torch.clip(time_vec, min=0.0, max=1.0)
    return t_samples


@torch.inference_mode()
def cm_sample_1_step(
        cm: torch.nn.Module, 
        n_samples: int, 
        data_shape: tuple, 
        device: str, 
        sigma_d: float,
        context: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    1st-order sampler by DDIM (1-step sampling)
    
    The formula is:
    x_t = cos(s-t)x_t - sin(s-t) sigma_d * F_theta(x_s/sigma_d, s)
    
    If we start from s=pi/2 (therefore x_s = z random noise) and t=0 we have
    x_0 = cos(pi/2)x_t - sin(pi/2) sigma_d * F_theta(z/sigma_d, pi/2)
        = -sigma_d * F_theta(z/sigma_d, pi/2)
    """
    # this is correct, the prior should be z ~ N(0, sigma_d x I)
    # since we calculate z / sigma_d this is the same as sampling
    # directly from N(0,I)
    z = torch.randn(n_samples, *data_shape).to(device)

    t = 1.56454 * torch.ones(n_samples, device=device)
    return -sigma_d * cm(z, t, context=context)
