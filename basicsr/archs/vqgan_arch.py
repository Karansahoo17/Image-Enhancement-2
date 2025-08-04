"""VQGAN Architecture components for CodeFormer"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import numpy as np

def normalize_weights(weight, fan_in):
    return weight / np.sqrt(fan_in)

class VectorQuantizer(nn.Module):
    """Vector Quantization layer for CodeFormer"""
    def __init__(self, codebook_size, emb_dim, beta=0.25):
        super().__init__()
        self.codebook_size = codebook_size  # number of embeddings
        self.emb_dim = emb_dim  # dimension of embedding
        self.beta = beta  # commitment cost used in loss term, beta * ||z_e(x)-sg[z_q(x)]||^2
        
        self.embedding = nn.Embedding(self.codebook_size, self.emb_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.emb_dim)
        
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.codebook_size).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        
        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        
        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss, {
            "perplexity": perplexity,
            "min_encodings": min_encodings,
            "min_encoding_indices": min_encoding_indices,
            "mean_distance": torch.mean(d)
        }
    
    def get_codebook_feat(self, indices, shape):
        # input indices: batch*token_num -> (batch*token_num)*1
        # shape: batch, height, width, channel
        indices = indices.view(-1,1)
        min_encodings = torch.zeros(indices.shape[0], self.codebook_size).to(indices)
        min_encodings.scatter_(1, indices, 1)
        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:  # reshape back to match original input shape
            z_q = z_q.view(shape).permute(0, 3, 1, 2).contiguous()

        return z_q

class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.GroupNorm(32, in_channel),
            nn.SiLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.GroupNorm(32, channel),
            nn.SiLU(),
            nn.Conv2d(channel, channel, 3, padding=1),
        )
        
        if in_channel != channel:
            self.skip = nn.Conv2d(in_channel, channel, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, input):
        out = self.conv(input)
        skip = self.skip(input)
        return out + skip

class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]
        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.GroupNorm(32, channel))
        blocks.append(nn.SiLU())

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)

class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.GroupNorm(32, channel))
        blocks.append(nn.SiLU())

        if stride == 4:
            blocks.extend([
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(channel, channel // 2, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(channel // 2, out_channel, 3, padding=1),
            ])
        elif stride == 2:
            blocks.extend([
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(channel, out_channel, 3, padding=1),
            ])

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)

class VQAutoEncoder(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
        loss_type='mse',
        latent_loss_weight=0.25,
        sample_minimization_weight=1.0,
        batch_maximization_weight=1.0,
    ):
        super().__init__()

        self.encoder = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.decoder = Decoder(embed_dim, in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=latent_loss_weight)

    def forward(self, input, return_recons_loss=False):
        quant, codebook_loss, quant_stats = self.encode(input)
        recons = self.decode(quant)
        
        if return_recons_loss:
            recons_loss = F.mse_loss(recons, input)
            return recons, codebook_loss, recons_loss, quant_stats
        else:
            return recons, codebook_loss, quant_stats

    def encode(self, input):
        enc = self.encoder(input)
        quant, codebook_loss, quant_stats = self.quantize(enc)
        return quant, codebook_loss, quant_stats

    def decode(self, quant):
        return self.decoder(quant)
