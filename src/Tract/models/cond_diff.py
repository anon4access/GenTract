from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor

# MONAI imports
from monai.networks.blocks import Convolution, TransformerBlock
from monai.utils import ensure_tuple_rep
from monai.networks.nets.diffusion_model_unet import get_down_block, get_timestep_embedding

import copy
from typing import Optional



##############################################
# Positional Encoding (batch-first)
##############################################
class PositionalEncoding(nn.Module):
    """
    Adds sinusoidal positional encodings to the input embeddings.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create a matrix of shape (max_len, d_model) with positional encodings.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        Returns:
            Tensor with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

##############################################
# Your Original Encoder (unchanged)
##############################################
class CoefficientEncoder(nn.Module):
    """
    Classification Network based on the Encoder of the Diffusion Model, followed by fully connected layers.
    This network is based on Wolleb et al. "Diffusion Models for Medical Anomaly Detection".
    (Note: No timestep embedding is used here.)
    
    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        num_res_blocks: number of residual blocks per level.
        channels: tuple of block output channels.
        attention_levels: list of levels to add attention.
        norm_num_groups: number of groups for normalization.
        norm_eps: epsilon for normalization.
        resblock_updown: if True, use residual blocks for downsampling.
        num_head_channels: number of channels in each attention head.
        with_conditioning: if True, add spatial transformers for conditioning.
        transformer_num_layers: number of transformer layers.
        cross_attention_dim: number of context dimensions.
        num_class_embeds: if specified, the model is class-conditional.
        upcast_attention: if True, upcast attention operations to full precision.
    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_res_blocks: Sequence[int] | int = (2, 2, 2, 2),
        channels: Sequence[int] = (32, 64, 64, 64),
        attention_levels: Sequence[bool] = (False, False, True, True),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        resblock_updown: bool = False,
        num_head_channels: int | Sequence[int] = 8,
        with_conditioning: bool = False,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        num_class_embeds: int | None = None,
        upcast_attention: bool = False,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
        model_dim: int = 32
    ) -> None:
        super().__init__()
        if with_conditioning is True and cross_attention_dim is None:
            raise ValueError("DiffusionModelEncoder expects cross_attention_dim when using with_conditioning.")
        if cross_attention_dim is not None and with_conditioning is False:
            raise ValueError("DiffusionModelEncoder expects with_conditioning=True when specifying cross_attention_dim.")
        if any((ch % norm_num_groups) != 0 for ch in channels):
            raise ValueError("All channel values must be multiples of norm_num_groups")
        if len(channels) != len(attention_levels):
            raise ValueError("channels and attention_levels must have the same length")
        if isinstance(num_head_channels, int):
            num_head_channels = ensure_tuple_rep(num_head_channels, len(attention_levels))
        if isinstance(num_res_blocks, int):
            num_res_blocks = ensure_tuple_rep(num_res_blocks, len(channels))
        if len(num_head_channels) != len(attention_levels):
            raise ValueError("num_head_channels must match the length of attention_levels.")

        self.in_channels = in_channels
        self.block_out_channels = channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_levels = attention_levels
        self.num_head_channels = num_head_channels
        self.with_conditioning = with_conditioning

        # Input convolution.
        # self.conv_in = Convolution(
        #     spatial_dims=spatial_dims,
        #     in_channels=in_channels,
        #     out_channels=channels[0],
        #     strides=1,
        #     kernel_size=3,
        #     padding=1,
        #     conv_only=True,
        # )

        # self.conv_out = Convolution(
        #     spatial_dims=spatial_dims,
        #     in_channels=channels[-1],
        #     out_channels=out_channels,
        #     strides=1,
        #     kernel_size=3,
        #     padding=1,
        #     conv_only=True
        # )


        # Class embedding dimension.
        class_emb_dim = model_dim
        self.num_class_embeds = num_class_embeds
        if num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, class_emb_dim)
        
        
        
        # Downsampling blocks.
        self.down_blocks = nn.ModuleList([])
        output_channel = in_channels
        for i in range(len(channels)):
            input_channel = output_channel
            output_channel = channels[i]
            is_final_block = i == len(channels)
            down_block = get_down_block(
                spatial_dims=spatial_dims,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=class_emb_dim,
                num_res_blocks=num_res_blocks[i],
                norm_num_groups=int(in_channels/2),
                norm_eps=norm_eps,
                add_downsample=not is_final_block,
                resblock_updown=resblock_updown,
                with_attn=(attention_levels[i] and not with_conditioning),
                with_cross_attn=(attention_levels[i] and with_conditioning),
                num_head_channels=num_head_channels[i],
                transformer_num_layers=transformer_num_layers,
                cross_attention_dim=cross_attention_dim,
                upcast_attention=upcast_attention,
                include_fc=include_fc,
                use_combined_linear=use_combined_linear,
                use_flash_attention=use_flash_attention,
            )
            self.down_blocks.append(down_block)
    #  self.out = nn.Sequential(nn.Linear(4096, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, self.out_channels))
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.num_class_embeds is not None:
            if class_labels is None:
                raise ValueError("class_labels must be provided when num_class_embeds > 0")
            class_emb = self.class_embedding(class_labels).to(dtype=x.dtype)
            emb = class_emb
        else:
            emb = 0.0
        #h = self.conv_in(x)
        h = x
        if context is not None and not self.with_conditioning:
            raise ValueError("with_conditioning must be True if context is provided")
        for down_block in self.down_blocks:
            h, _ = down_block(hidden_states=h, temb=emb, context=context)

        #h = self.conv_out(h)

        return h

##############################################
# Alias for clarity: our encoder is now CoefficientEncoder
##############################################

##############################################
# MultiCoefficientEncoder
##############################################
class MultiCoefficientEncoder(nn.Module):
    """
    Wraps a CoefficientEncoder to process M coefficient volumes.
    Expects input of shape (B, M, 1, H, W, D) and outputs a latent tensor of shape
    (B, M, latent_channels, H_lat, W_lat, D_lat).
    """
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, M, _, H, W, D = x.shape
        latent_list = []
        for i in range(M):
            xi = x[:, i, ...]  # (B, 1, H, W, D)
            class_labels = torch.full((B,), i, device=x.device, dtype=torch.long)
            latent_i = self.encoder(xi, class_labels=class_labels)
            latent_list.append(latent_i)
        latent = torch.stack(latent_list, dim=1)
        return latent

##############################################
# LatentSelfAttention with CLS Token (kept in output)
##############################################
class LatentSelfAttention(nn.Module):
    """
    Applies self-attention over the latent space.
    Inserts a learnable CLS token at the beginning of the flattened latent sequence to help
    extract a global coefficient representation. The CLS token is kept in the final output sequence.
    
    Input: (B, M, C, H, W, D)
    Output: A sequence tensor of shape (B, L, M*C), where L = H*W*D + 1.
    
    Args:
        latent_channels (int): Number of channels (C) in each latent.
        M (int): Number of coefficient volumes.
        spatial_dims (int): Number of spatial dimensions.
        num_transformer_layers (int): Number of transformer layers.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate.
    """
    def __init__(self, latent_channels: int, M: int, spatial_dims: int, num_transformer_layers: int = 1, num_heads: int = 8, dropout: float = 0.0, use_flash_attention=False):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.M = M
        self.latent_channels = latent_channels
        self.d_model = M * latent_channels
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=self.d_model,
                mlp_dim=self.d_model * 4,
                num_heads=num_heads,
                dropout_rate=dropout,
                with_cross_attention=False,
                use_flash_attention=use_flash_attention,
            ) for _ in range(num_transformer_layers)
        ])
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, M, C, H, W, D = x.shape
        seq_len = H * W * D
        x_flat = x.view(B, M * C, seq_len).permute(0, 2, 1).contiguous()  # (B, seq_len, M*C)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, M*C)
        x_cat = torch.cat([cls_tokens, x_flat], dim=1)  # (B, seq_len+1, M*C)
        for block in self.transformer_blocks:
            x_cat = block(x_cat)
        return x_cat  # (B, seq_len+1, M*C)

##############################################
# LatentContextProjection
##############################################
class LatentContextProjection(nn.Module):
    """
    Projects a latent sequence into a context for cross-attention.
    Expects input of shape (B, L, M*C) where L = H*W*D + 1, and outputs (B, L, model_dim).
    
    Args:
        M (int): Number of coefficient volumes.
        latent_channels (int): Number of channels per latent.
        model_dim (int): Desired model dimension.
    """
    def __init__(self, M: int, latent_channels: int, model_dim: int):
        super().__init__()
        self.projector = nn.Linear(M * latent_channels, model_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = self.projector(x)
        return context




class TransformerEncoderLayerWithConditioning(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        # Initialize the parent class (Torch's TransformerEncoderLayer)
        super().__init__(*args, **kwargs)
        d_model = self.self_attn.embed_dim
        nhead = self.self_attn.num_heads
        # Add a cross-attention module for conditioning.
        # We use the same dropout value and batch_first setting as self.self_attn.
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=self.dropout.p, batch_first=self.self_attn.batch_first
        )
        # Extra normalization layer for the cross-attention branch.
        self.norm_cross = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, context=None, is_causal=False):
        """
        Args:
            src: Input tensor (batch, seq, d_model) if batch_first=True.
            src_mask, src_key_padding_mask: Same as in the base class.
            context: Optional conditioning tensor (batch, cond_seq, d_model).
            is_causal: Whether to use a causal mask for self-attention.
            
        Returns:
            Tensor of shape (batch, seq, d_model).
            
        Ordering: self-attention, then cross-attention (if context is provided), then feed-forward.
        """

        # Note: Overriding forward disables the fast path.
        if self.norm_first:
            # Pre-norm formulation:
            x = src
            # Self-attention block.
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal)
            # Cross-attention block (if conditioning is provided).
            if context is not None:
                x = x + self.cross_attn(self.norm_cross(x), context, context, need_weights=False)[0]
            # Feed-forward block.
            x = x + self._ff_block(self.norm2(x))
        else:
            # Post-norm formulation:
            # Self-attention block.
            x = self.norm1(src + self._sa_block(src, src_mask, src_key_padding_mask, is_causal))
            # Cross-attention block.
            if context is not None:
                x = self.norm_cross(x + self.cross_attn(x, context, context, need_weights=False)[0])
            # Feed-forward block.
            x = self.norm2(x + self._ff_block(x))
        return x



def _get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class TransformerEncoderWithContext(nn.Module):
    """
    A wrapper that builds an encoder from layers that support a `context` argument.
    
    Args:
        encoder_layer: An encoder layer that supports an extra `context` argument in its forward() method.
        num_layers: Number of layers to clone.
        norm: Optional normalization module applied after all layers. If not provided,
              a default LayerNorm using encoder_layer.self_attn.embed_dim will be created.
    """
    def __init__(
        self,
        encoder_layer: nn.Module,
        num_layers: int,
        norm: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        if norm is None:
            # Create a default norm using the encoder layer's embed_dim if available.
            try:
                d_model = encoder_layer.self_attn.embed_dim
            except AttributeError:
                raise ValueError("encoder_layer does not have attribute self_attn.embed_dim; please provide a norm.")
            norm = nn.LayerNorm(d_model)
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        is_causal: Optional[bool] = False,
    ) -> Tensor:
        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
                context=context,
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


##############################################
# DiffusionTransformer (no CLS token for diffusion input)
##############################################
class DiffusionTransformer(nn.Module):
    """
    Diffusion Transformer that processes a sequential diffusion input with positional encoding
    and applies cross-attention using a latent context computed from raw coefficient volumes.
    
    In the forward pass, the raw context (of shape (B, M, 1, H, W, D)) is processed via:
      1. A MultiCoefficientEncoder to produce latents of shape (B, M, C, H_lat, W_lat, D_lat).
      2. A latent self-attention block (with a CLS token) that outputs a sequence of shape (B, L, M*C),
         where L = H_lat*W_lat*D_lat + 1.
      3. A projection (LatentContextProjection) to map the sequence to (B, L, model_dim).
    Simultaneously, the diffusion input (of shape (B, num_points, input_dim)) is embedded, has positional encoding applied,
    and then processed through transformer blocks with cross-attention using the computed latent context.
    The output is then projected back to (B, num_points, input_dim).
    
    Args:
        num_points (int): Sequence length for diffusion input.
        input_dim (int): Dimensionality of diffusion input tokens.
        model_dim (int): Transformer model dimension.
        num_transformer_layers (int): Number of transformer layers.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate.
        M (int): Number of coefficient volumes.
        latent_channels (int): Number of channels in each latent representation.
    """
    def __init__(
        self, 
        input_dim: int = 3, 
        num_points = 128,
        encoder_channels: Sequence = (16, 32, 64, 128),
        res_blocks: Sequence[int] | int = (2, 2, 2, 2),
        encoder_att_levels:  Sequence[bool] = (False, False, False, False),
        norm_num_groups: int = 4,
        norm_eps: float = 1e-6,
        sab_layers: int = 1,
        sab_heads: int = 4,
        model_dim: int = 512, 
        diff_transformer_layers: int = 4, 
        diff_num_heads: int = 8, 
        dropout: float = 0.1, 
        M: int = 28,
        use_flash_attention = False,
        model_type='encoder',
        cond=False,
        ):
        
        print('Running model with conditioning: ', cond)

        latent_channels = encoder_channels[-1]
        
        super().__init__()

        self.M = M 
        self.num_points = num_points

        self.model_dim = model_dim
        self.input_embed = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.ReLU())
        
        self.positional_encoding = PositionalEncoding(model_dim, 
                                                      dropout=dropout, 
                                                      max_len=num_points)
    ##=======================================================
    # ONLY USED WHEN CONDITIONING MODEL ON LATENTS
        if cond:

            print('Initialising condititioning model with num coeffs ', M)
            
            self.encoder_chans = encoder_channels
            self.res_blocks = res_blocks
            self.encoder_att_levels = encoder_att_levels
            self.latent_chans = latent_channels

            self.sab_layers = sab_layers
            self.sab_heads = sab_heads


            self.model_type = model_type

            self.coeff_encoder = CoefficientEncoder(
                spatial_dims=3,
                in_channels=4,
                out_channels=latent_channels,
                num_res_blocks=res_blocks,
                channels=encoder_channels,
                attention_levels=encoder_att_levels,
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                resblock_updown=False,
                num_head_channels=8,
                with_conditioning=False,
                transformer_num_layers=1,
                cross_attention_dim=None,
                num_class_embeds=M,
                use_flash_attention=use_flash_attention,
                model_dim=model_dim
            )

            self.multi_encoder = MultiCoefficientEncoder(encoder=self.coeff_encoder)


            if self.sab_layers > 0: 

                self.latent_self_attn = LatentSelfAttention(
                    latent_channels=latent_channels,    
                    M=M, 
                    spatial_dims=3, 
                    num_transformer_layers=sab_layers, 
                    num_heads=sab_heads, 
                    dropout=0.1,
                    use_flash_attention=False)
                
            else:

                self.latent_self_attn = None

            self.latent_context_proj = LatentContextProjection(
                M=M, 
                latent_channels=latent_channels, 
                model_dim=model_dim)
            

    ##=======================================================

        self.diff_transformer_layers = diff_transformer_layers
        self.diff_num_heads = diff_num_heads

        encoder_layer = TransformerEncoderLayerWithConditioning(
                d_model=model_dim,
                nhead=diff_num_heads,
                dropout=0.1,
                batch_first=True,
            )
        
        self.encoder = TransformerEncoderWithContext(encoder_layer, num_layers=diff_transformer_layers)

        self.dropout = dropout
        # Project the encoder output back to the original 3D coordinate space.
        self.output_proj = nn.Linear(model_dim, input_dim)
        # Time embedding MLP.
        self.time_embed = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.SiLU(),
            nn.Linear(model_dim * 4, model_dim)
        )




    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Diffusion input tensor of shape (B, num_points, input_dim)
            context: Raw coefficient volumes of shape (B, M, 1, H, W, D)
        Returns:
            Predicted noise tensor of shape (B, num_points, input_dim)
        """

        if timesteps.ndim == 0:
            timesteps = timesteps.view(1)    

        t_emb = get_timestep_embedding(timesteps, self.model_dim)
        t_emb = t_emb.to(dtype=x.dtype)
        t_emb = self.time_embed(t_emb)

        if context is not None:
            # Process raw context.
            if self.encoder_chans[0] == 4:
                print('Not using class-conditioned encoder')
                latent = context
            else:
                latent = self.multi_encoder(context)  # (B, M, C, H_lat, W_lat, D_lat)
            
            if self.latent_self_attn is not None:
                latent = self.latent_self_attn(latent)  # (B, seq_len+1, M*C) where seq_len = H_lat*W_lat*D_lat + 1 (includes CLS token)
            else:
                B, M, C, H_lat, W_lat, D_lat = latent.shape
                latent = latent.contiguous().view(B, M * C, H_lat * W_lat * D_lat)
        #        Transpose to obtain shape (B, H_lat*W_lat*D_lat, M*C)
                latent = latent.transpose(1, 2)            

            latent_context = self.latent_context_proj(latent)  # (B, seq_len+1, model_dim)
            
        else:

            latent_context = None

        # Process diffusion input.
        x_embed = self.input_embed(x) * math.sqrt(self.model_dim)  # (B, num_points, model_dim)
        x_embed = x_embed + t_emb.unsqueeze(1)


        x_embed = self.positional_encoding(x_embed)  # (B, num_points, model_dim)

        ## unsure if needed or of CA/einops does this
        if latent_context is not None:

            latent_context = latent_context.expand(x.shape[0], -1, -1) # expansion of conditioning across batch (multiple stramlines same subject)


        x_embed = self.encoder(x_embed, context=latent_context)

        out = self.output_proj(x_embed)

        return out
    
    
##############################################
# A model wrapper to use within ODE solver.
##############################################
 
class VelocityModelWrapper(nn.Module):
    """
    A corrected wrapper that handles both latent context,
    matching the model's training signature.
    """
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, context) -> torch.Tensor:
        """
        """
        # If context is a dictionary, it contains both latent 
        if isinstance(context, dict):
            latent_context = context.get("latent")
            return self.model(x=x, timesteps=t, context=latent_context)
        else:
            # Fallback for cases where only latent context is passed 
            return self.model(x=x, timesteps=t, context=context)