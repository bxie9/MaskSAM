# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from typing import Type
from einops import rearrange
    
class AdapterTFusionMiddleMLPTQReshape(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, num_frames=4, num_queries=8, t_mlp_ratio=4, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

        self.num_frames = num_frames
        self.num_queries = num_queries
        T_hidden_features = int(num_frames * t_mlp_ratio)
        self.act = act_layer()
        self.T_fc1 = nn.Linear(num_frames, T_hidden_features)
        self.T_fc2 = nn.Linear(T_hidden_features, num_frames)
        
    def forward(self, x):

        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)

        # x is (BT, HW+1, D)
        if len(xs.shape) == 4:
            xt = rearrange(xs, '(b d q) h w c -> (b q) h w c d', d=self.num_frames, q=self.num_queries)
        else:
            xt = rearrange(xs, '(b d q) hw c -> (b q) hw c d', d=self.num_frames, q=self.num_queries)
        
        xt = self.T_fc1(xt.contiguous())
        xt = self.act(xt)
        xt = self.T_fc2(xt)
        if len(x.shape) == 4:
            xt = rearrange(xt, '(b q) h w c d -> (b d q) h w c', q=self.num_queries)
        else:
            xt = rearrange(xt, '(b q) hw c d -> (b d q) hw c', q=self.num_queries)
        
        # skip connection
        xs = xs + xt

        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x

class AdapterTFusion3DMiddleDWConvNorm(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, num_frames=4, act_layer=nn.GELU, skip_connect=True, size=(64, 64)):
        super().__init__()
        self.skip_connect = skip_connect
        self.num_frames = num_frames
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

        self.size = size

        self.T_DWconv = nn.Conv3d(in_channels=D_hidden_features, out_channels=D_hidden_features,
                    kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1,1,1), groups=D_hidden_features)
        self.norm = LayerNorm3d(D_hidden_features)
        
    def forward(self, x):

        xs = self.D_fc1(x)
        xs = self.act(xs)

        if len(xs.shape) == 4:
            xt = rearrange(xs, '(b d) h w c -> b c d h w', d=self.num_frames)
        else:
            xt = rearrange(xs, '(b d) (h w) c -> b c d h w', d=self.num_frames,  w=self.size[1])
        
        xt = self.T_DWconv(xt.contiguous())
        xt = self.norm(xt)
        xt = self.act(xt)
        if len(x.shape) == 4:
            xt = rearrange(xt, 'b c d h w -> (b d) h w c')
        else:
            xt = rearrange(xt, 'b c d h w -> (b d) (h w) c')
        
        xs = xs + xt

        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x
    
class AdapterTFusion3DMiddleDWConvNormTQReshape(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, num_frames=4, num_queries=8, act_layer=nn.GELU, skip_connect=True, size=(64, 64)):
        super().__init__()
        self.skip_connect = skip_connect
        self.num_frames = num_frames
        self.num_queries = num_queries
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

        self.size = size

        self.T_DWconv = nn.Conv3d(in_channels=D_hidden_features, out_channels=D_hidden_features,
                    kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1,1,1), groups=D_hidden_features)
        self.norm = LayerNorm3d(D_hidden_features)
        
    def forward(self, x):

        xs = self.D_fc1(x)
        xs = self.act(xs)

        if len(xs.shape) == 4:
            xt = rearrange(xs, '(b d q) h w c -> (b q) c d h w', d=self.num_frames, q=self.num_queries)
        else:
            xt = rearrange(xs, '(b d q) (h w) c -> (b q) c d h w', d=self.num_frames, h=self.size[0], w=self.size[1], q=self.num_queries)
        
        xt = self.T_DWconv(xt.contiguous())
        xt = self.norm(xt)
        xt = self.act(xt)
        if len(x.shape) == 4:
            xt = rearrange(xt, '(b q) c d h w -> (b d q) h w c', q=self.num_queries)
        else:
            xt = rearrange(xt, '(b q) c d h w -> (b d q) (h w) c', q=self.num_queries)
        
        xs = xs + xt

        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class LayerNorm3d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x