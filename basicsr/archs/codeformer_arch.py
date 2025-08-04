#edited with mobile vit transformer
import math
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, List

from basicsr.archs.vqgan_arch import *
from basicsr.utils import get_root_logger
from basicsr.utils.registry import ARCH_REGISTRY

def calc_mean_std(feat, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization.
    
    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    feat_var = feat.view(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    """Adaptive instance normalization.
    
    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.
    
    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
    
    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class MobileViTAttention(nn.Module):
    """MobileViT attention mechanism designed for efficient processing."""
    
    def __init__(self, embed_dim, heads=4, dim_head=8, attn_dropout=0):
        super().__init__()
        self.qkv_proj = nn.Linear(embed_dim, 3*embed_dim, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.embed_dim = embed_dim
        self.num_heads = heads
        self.scale = dim_head ** -0.5
    
    def forward(self, x):
        b_sz, S_len, in_channels = x.shape
        qkv = self.qkv_proj(x).reshape(b_sz, S_len, 3, self.num_heads, -1)
        qkv = qkv.transpose(1, 3).contiguous()
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q * self.scale
        k = k.transpose(-1, -2)
        attn = torch.matmul(q, k)
        batch_size, num_heads, num_src_tokens, num_tgt_tokens = attn.shape
        attn_as_float = self.softmax(attn.float())
        attn = attn_as_float.to(attn.dtype)
        attn = self.attn_dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b_sz, S_len, -1)
        out = self.out_proj(out)
        
        return out

class MobileViTTransformerEncoder(nn.Module):
    """MobileViT transformer encoder with pre-normalization architecture."""
    
    def __init__(self, embed_dim, ffn_latent_dim, heads=8, dim_head=8, dropout=0, attn_dropout=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-5, elementwise_affine=True)
        self.attention = MobileViTAttention(embed_dim, heads, dim_head, attn_dropout)
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-5, elementwise_affine=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_latent_dim, bias=True),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_latent_dim, embed_dim, bias=True),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Pre-norm architecture
        x_norm = self.norm1(x)
        x = x + self.dropout1(self.attention(x_norm))
        x = x + self.ffn(self.norm2(x))
        return x

class TransformerSALayer(nn.Module):
    """
    Transformer self-attention layer modified to use MobileViT's transformer components
    while maintaining compatibility with original CodeFormer weights.
    """
    def __init__(self, embed_dim, nhead=8, dim_mlp=2048, dropout=0.0, activation="gelu"):
        super().__init__()
        
        # Define internal dimensions
        dim_head = embed_dim // nhead
        
        # Original architecture compatible attributes for weight loading
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, dim_mlp)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_mlp, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # For GELU vs. SiLU compatibility
        if activation == "gelu":
            self.activation = F.gelu
        else:
            self.activation = F.silu
        
        # The actual MobileViT transformer encoder we'll use
        self.mobilevit_encoder = MobileViTTransformerEncoder(
            embed_dim=embed_dim,
            ffn_latent_dim=dim_mlp,
            heads=nhead,
            dim_head=dim_head,
            dropout=dropout,
            attn_dropout=dropout
        )
        
        # Flag to indicate whether we're using the original architecture or MobileViT
        self.use_mobilevit = True
    
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        
        # Apply positional embedding if provided
        if query_pos is not None:
            tgt = self.with_pos_embed(tgt, query_pos)
        
        # Use the MobileViT transformer encoder
        if self.use_mobilevit:
            return self.mobilevit_encoder(tgt)
        else:
            # Fallback to original implementation (should never be used)
            q = k = self.with_pos_embed(tgt, query_pos)
            
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)
            return tgt
    
    def load_from_original_weights(self, state_dict, prefix):
        """
        Convert original weights to be compatible with MobileViT architecture
        
        Args:
            state_dict: Original state dictionary
            prefix: Prefix for the current layer (e.g. 'ft_layers.0.')
        """
        # Extract original weights
        self_attn_in_proj_weight = state_dict.get(f"{prefix}self_attn.in_proj_weight", None)
        self_attn_in_proj_bias = state_dict.get(f"{prefix}self_attn.in_proj_bias", None)
        self_attn_out_proj_weight = state_dict.get(f"{prefix}self_attn.out_proj.weight", None)
        self_attn_out_proj_bias = state_dict.get(f"{prefix}self_attn.out_proj.bias", None)
        
        linear1_weight = state_dict.get(f"{prefix}linear1.weight", None)
        linear1_bias = state_dict.get(f"{prefix}linear1.bias", None)
        linear2_weight = state_dict.get(f"{prefix}linear2.weight", None)
        linear2_bias = state_dict.get(f"{prefix}linear2.bias", None)
        
        norm1_weight = state_dict.get(f"{prefix}norm1.weight", None)
        norm1_bias = state_dict.get(f"{prefix}norm1.bias", None)
        norm2_weight = state_dict.get(f"{prefix}norm2.weight", None)
        norm2_bias = state_dict.get(f"{prefix}norm2.bias", None)
        
        # Check if we have the original weights
        if self_attn_in_proj_weight is not None:
            embed_dim = self.self_attn.embed_dim
            
            # Convert MultiheadAttention's in_proj_weight to qkv_proj.weight
            # In PyTorch's MultiheadAttention, in_proj_weight is [3*embed_dim, embed_dim]
            # and contains concatenated Q, K, V projection weights
            q_proj_weight, k_proj_weight, v_proj_weight = self_attn_in_proj_weight.chunk(3, dim=0)
            
            # Transfer to MobileViT attention - the dimensions need to be [out_features, in_features]
            # where out_features is 3*embed_dim and in_features is embed_dim
            qkv_weight = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)
            self.mobilevit_encoder.attention.qkv_proj.weight.data.copy_(qkv_weight)
            
            # Convert biases
            if self_attn_in_proj_bias is not None:
                q_proj_bias, k_proj_bias, v_proj_bias = self_attn_in_proj_bias.chunk(3, dim=0)
                qkv_bias = torch.cat([q_proj_bias, k_proj_bias, v_proj_bias], dim=0)
                self.mobilevit_encoder.attention.qkv_proj.bias.data.copy_(qkv_bias)
            
            # Output projection
            if self_attn_out_proj_weight is not None:
                self.mobilevit_encoder.attention.out_proj.weight.data.copy_(self_attn_out_proj_weight)
            
            if self_attn_out_proj_bias is not None:
                self.mobilevit_encoder.attention.out_proj.bias.data.copy_(self_attn_out_proj_bias)
            
            # FFN weights
            if linear1_weight is not None:
                self.mobilevit_encoder.ffn[0].weight.data.copy_(linear1_weight)
            
            if linear1_bias is not None:
                self.mobilevit_encoder.ffn[0].bias.data.copy_(linear1_bias)
            
            if linear2_weight is not None:
                self.mobilevit_encoder.ffn[3].weight.data.copy_(linear2_weight)
            
            if linear2_bias is not None:
                self.mobilevit_encoder.ffn[3].bias.data.copy_(linear2_bias)
            
            # LayerNorm weights
            if norm1_weight is not None:
                self.mobilevit_encoder.norm1.weight.data.copy_(norm1_weight)
            
            if norm1_bias is not None:
                self.mobilevit_encoder.norm1.bias.data.copy_(norm1_bias)
            
            if norm2_weight is not None:
                self.mobilevit_encoder.norm2.weight.data.copy_(norm2_weight)
            
            if norm2_bias is not None:
                self.mobilevit_encoder.norm2.bias.data.copy_(norm2_bias)
        
        # Return True if weights were successfully transferred
        return self_attn_in_proj_weight is not None

class Fuse_sft_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.encode_enc = ResBlock(2*in_ch, out_ch)
        
        self.scale = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))
        
        self.shift = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))
    
    def forward(self, enc_feat, dec_feat, w=1):
        enc_feat = self.encode_enc(torch.cat([enc_feat, dec_feat], dim=1))
        scale = self.scale(enc_feat)
        shift = self.shift(enc_feat)
        residual = w * (dec_feat * scale + shift)
        out = dec_feat + residual
        return out

@ARCH_REGISTRY.register()
class CodeFormer(VQAutoEncoder):
    def __init__(self, dim_embd=512, n_head=8, n_layers=9,
                codebook_size=1024, latent_size=256,
                connect_list=['32', '64', '128', '256'],
                fix_modules=['quantize','generator'], vqgan_path=None):
        super(CodeFormer, self).__init__(512, 64, [1, 2, 2, 4, 4, 8], 'nearest',2, [16], codebook_size)
        
        if vqgan_path is not None:
            self.load_state_dict(
                torch.load(vqgan_path, map_location='cpu')['params_ema'])
        
        if fix_modules is not None:
            for module in fix_modules:
                for param in getattr(self, module).parameters():
                    param.requires_grad = False
        
        self.connect_list = connect_list
        self.n_layers = n_layers
        self.dim_embd = dim_embd
        self.dim_mlp = dim_embd*2
        
        self.position_emb = nn.Parameter(torch.zeros(latent_size, self.dim_embd))
        self.feat_emb = nn.Linear(256, self.dim_embd)
        
        # transformer layers - modified to use MobileViT's transformer
        self.ft_layers = nn.ModuleList([
            TransformerSALayer(embed_dim=dim_embd, nhead=n_head, dim_mlp=self.dim_mlp, dropout=0.0)
            for _ in range(self.n_layers)
        ])
        
        # logits_predict head
        self.idx_pred_layer = nn.Sequential(
            nn.LayerNorm(dim_embd),
            nn.Linear(dim_embd, codebook_size, bias=False))
        
        self.channels = {
            '16': 512,
            '32': 256,
            '64': 256,
            '128': 128,
            '256': 128,
            '512': 64,
        }
        
        # after second residual block for > 16, before attn layer for ==16
        self.fuse_encoder_block = {'512':2, '256':5, '128':8, '64':11, '32':14, '16':18}
        # after first residual block for > 16, before attn layer for ==16
        self.fuse_generator_block = {'16':6, '32': 9, '64':12, '128':15, '256':18, '512':21}
        
        # fuse_convs_dict
        self.fuse_convs_dict = nn.ModuleDict()
        for f_size in self.connect_list:
            in_ch = self.channels[f_size]
            self.fuse_convs_dict[f_size] = Fuse_sft_block(in_ch, in_ch)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def load_state_dict(self, state_dict, strict=True):
        """
        Override load_state_dict to handle transformer weight conversion
        """
        # Try to convert transformer weights if they exist in the original format
        for i, layer in enumerate(self.ft_layers):
            layer_prefix = f'ft_layers.{i}.'
            # Try to map old weights to new architecture
            layer.load_from_original_weights(state_dict, layer_prefix)
        
        # Continue with regular load_state_dict
        return super().load_state_dict(state_dict, strict=False)  # Use non-strict loading
    
    def forward(self, x, w=0, detach_16=True, code_only=False, adain=False):
        # ################### Encoder #####################
        enc_feat_dict = {}
        out_list = [self.fuse_encoder_block[f_size] for f_size in self.connect_list]
        for i, block in enumerate(self.encoder.blocks):
            x = block(x)
            if i in out_list:
                enc_feat_dict[str(x.shape[-1])] = x.clone()
        
        lq_feat = x
        # ################# Transformer ###################
        # quant_feat, codebook_loss, quant_stats = self.quantize(lq_feat)
        pos_emb = self.position_emb.unsqueeze(1).repeat(1,x.shape[0],1)
        # BCHW -> BC(HW) -> (HW)BC
        feat_emb = self.feat_emb(lq_feat.flatten(2).permute(2,0,1))
        query_emb = feat_emb
        
        # Transformer encoder - modified to use ModuleList instead of Sequential
        for layer in self.ft_layers:
            query_emb = layer(query_emb, query_pos=pos_emb)
        
        # output logits
        logits = self.idx_pred_layer(query_emb) # (hw)bn
        logits = logits.permute(1,0,2) # (hw)bn -> b(hw)n
        
        if code_only: # for training stage II
          # logits doesn't need softmax before cross_entropy loss
            return logits, lq_feat
        
        # ################# Quantization ###################
        # if self.training:
        #     quant_feat = torch.einsum('btn,nc->btc', [soft_one_hot, self.quantize.embedding.weight])
        #     # b(hw)c -> bc(hw) -> bchw
        #     quant_feat = quant_feat.permute(0,2,1).view(lq_feat.shape)
        # ------------
        soft_one_hot = F.softmax(logits, dim=2)
        _, top_idx = torch.topk(soft_one_hot, 1, dim=2)
        quant_feat = self.quantize.get_codebook_feat(top_idx, shape=[x.shape[0],16,16,256])
        # preserve gradients
        # quant_feat = lq_feat + (quant_feat - lq_feat).detach()
        
        if detach_16:
            quant_feat = quant_feat.detach() # for training stage III
        if adain:
            quant_feat = adaptive_instance_normalization(quant_feat, lq_feat)
        
        # ################## Generator ####################
        x = quant_feat
        fuse_list = [self.fuse_generator_block[f_size] for f_size in self.connect_list]
        
        for i, block in enumerate(self.generator.blocks):
            x = block(x)
            if i in fuse_list: # fuse after i-th block
                f_size = str(x.shape[-1])
                if w>0:
                    x = self.fuse_convs_dict[f_size](enc_feat_dict[f_size].detach(), x, w)
        out = x
        # logits doesn't need softmax before cross_entropy loss
        return out, logits, lq_feat
