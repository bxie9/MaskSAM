import argparse
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
# import monai
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from typing import Any, Iterable
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split
from einops import rearrange, repeat
import scipy
from torch.nn import functional as F
import fvcore.nn.weight_init as weight_init

from .segment_anything import sam_model_registry
from .segment_anything.utils.transforms import ResizeLongestSide

from typing import Union, Type, List, Tuple

import torchvision
import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
from torchvision.transforms.functional import resize
from torchvision.ops import masks_to_boxes
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
import random

class TestDecoder(nn.Module):

    def __init__(
        self,
        deep_supervision=True):
        super().__init__()

        self.deep_supervision = deep_supervision

    def forward(self, x):
        return x

class Conv3d(torch.nn.Conv3d):
    """
    A wrapper around :class:`torch.nn.Conv3d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv3d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv3d` in these PyTorch versions has already supported empty inputs.

        x = F.conv3d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

####### Mask SAM ############
class SAMAdapter_2024_AMOS_SmallPatch_FinalLayerCNN_TQReshape_Patch256(nn.Module):

    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        nonlin_first: bool = False,
        return_skips: bool = True,
        pool: str = 'conv',
        device: str = 'cuda',
        model_type = "vit_h_adapter_2024_acdc_smallpatch_tqreshape", #""vit_b",
        checkpoint = "/data/xiebin/nnunet_tangseng/nnUNet/nnunetv2/sam/sam_checkpoints/sam_vit_h_4b8939.pth",
        # model_type = "vit_b_adapter_2024_acdc_smallpatch_tqreshape", #""vit_b",
        # checkpoint = "/data/xiebin/nnunet/nnUNet/nnunetv2/sam/sam_checkpoints/sam_vit_b_01ec64.pth",
        frames: int=4,
        num_queries: int=24,
        # checkpoint: str = "/data/xiebin/nnSeries/nnunetv2/nnUNet/nnunetv2/sam/sam_checkpoints/sam_vit_b_01ec64.pth",
    ):
        super().__init__()

        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert len(kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                             "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"

        # self.stages = nn.Sequential(*stages)
        print('=============', num_classes)
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.input_channels = input_channels
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # we store some things that a potential decoder needs
        self.n_stages = n_stages
        self.conv_op = conv_op = torch.nn.Conv3d
        self.norm_op = norm_op = torch.nn.InstanceNorm3d
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin = torch.nn.LeakyReLU
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

        self.device = device
        self.sam_checkpoint_dir = checkpoint
        self.model_type = model_type

        self.num_queries = num_queries

        # get the model
        self.sam_model = sam_model_registry[self.model_type](
            checkpoint=self.sam_checkpoint_dir,num_frames=frames, num_classes=num_classes, num_queries=num_queries
        )
        # model.to(self.device)

        self.sam_trans = ResizeLongestSide(256)
        
        self.decoder = TestDecoder(deep_supervision)


        self.first_conv = nn.Sequential(
            StackedConvBlocks(1, conv_op, input_channels, 8, (3, 3, 3), 1, conv_bias, norm_op, norm_op_kwargs,
                dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first),
            StackedConvBlocks(1, conv_op, 8, 3, (3, 3, 3), 1, conv_bias, norm_op, norm_op_kwargs,
                dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first),
        )
        
        # we start with the bottleneck and work out way up
        image_decoder_stages = []
        # image_decoder_stages_first = []
        image_decoder_transpconvs = []
        image_decoder_seg_layers = []
        image_decoder_skips = []
        # image_decoder_skips_first = []
        # image_decoder_features_per_stage = features_per_stage[:5]
        # image_decoder_strides = strides
        image_decoder_features_per_stage = [64, 64, 128, 128, 256, 256]
        image_decoder_strides = [[1, 2, 2], [1, 2, 2], [1, 2, 2], [1, 2, 2], [1, 2, 2], [1, 1, 1]]
        image_decoder_n_stages = 6
        
        bbox_conv_input_features = 0
        for s in range(1, image_decoder_n_stages):
            input_features_below = image_decoder_features_per_stage[-s]
            input_features_skip = image_decoder_features_per_stage[-(s + 1)]
            stride_for_transpconv = image_decoder_strides[-s]
            if s == 1:
                # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
                image_decoder_stages.append(StackedConvBlocks(
                    n_conv_per_stage[s-1], conv_op, input_features_below, input_features_skip,
                    kernel_sizes[-(s + 1)], 1, conv_bias, norm_op, norm_op_kwargs,
                    dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
                ))
            elif s == image_decoder_n_stages - 1:
                image_decoder_transpconvs.append(nn.ConvTranspose3d(
                    input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                    bias=conv_bias
                ))
                # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
                image_decoder_stages.append(StackedConvBlocks(
                    n_conv_per_stage[s-1]-1, conv_op, input_features_skip*2, input_features_skip,
                    kernel_sizes[-(s + 1)], 1, conv_bias, norm_op, norm_op_kwargs,
                    dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
                ))
                image_decoder_skips_temp = [nn.ConvTranspose3d(
                    self.sam_model.image_encoder.embed_dim, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                    bias=conv_bias,),
                    StackedConvBlocks(
                    n_conv_per_stage[s-1]-1, conv_op, input_features_skip, input_features_skip,
                    kernel_sizes[-(s + 1)], 1, conv_bias, norm_op, norm_op_kwargs,
                    dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first)]
                for s_i in range(s-2): 
                    image_decoder_skips_temp.append(nn.ConvTranspose3d(
                    input_features_skip, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                    bias=conv_bias,))
                    image_decoder_skips_temp.append(StackedConvBlocks(
                    n_conv_per_stage[s-1]-1, conv_op, input_features_skip, input_features_skip,
                    kernel_sizes[-(s + 1)], 1, conv_bias, norm_op, norm_op_kwargs,
                    dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first))
                image_decoder_skips.append(nn.Sequential(
                    *image_decoder_skips_temp
                ))

            else:
                image_decoder_transpconvs.append(nn.ConvTranspose3d(
                    input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                    bias=conv_bias
                ))
                # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
                image_decoder_stages.append(StackedConvBlocks(
                    n_conv_per_stage[s-1]-1, conv_op, input_features_skip*2, input_features_skip,
                    kernel_sizes[-(s + 1)], 1, conv_bias, norm_op, norm_op_kwargs,
                    dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
                ))
                image_decoder_skips_temp = [nn.ConvTranspose3d(
                    self.sam_model.image_encoder.embed_dim, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                    bias=conv_bias,),
                    StackedConvBlocks(
                    n_conv_per_stage[s-1]-1, conv_op, input_features_skip, input_features_skip,
                    kernel_sizes[-(s + 1)], 1, conv_bias, norm_op, norm_op_kwargs,
                    dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first)]
                for s_i in range(s-2):
                    image_decoder_skips_temp.append(nn.ConvTranspose3d(
                    input_features_skip, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                    bias=conv_bias,))
                    image_decoder_skips_temp.append(StackedConvBlocks(
                    n_conv_per_stage[s-1]-1, conv_op, input_features_skip, input_features_skip,
                    kernel_sizes[-(s + 1)], 1, conv_bias, norm_op, norm_op_kwargs,
                    dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first))
                image_decoder_skips.append(nn.Sequential(
                    *image_decoder_skips_temp
                ))
                # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
                # then a model trained with deep_supervision=True could not easily be loaded at inference time where
                # deep supervision is not needed. It's just a convenience thing
            
            bbox_conv_input_features += input_features_skip
            if s == image_decoder_n_stages-1:
                image_decoder_seg_layers.append(conv_op(input_features_skip, num_queries, 1, 1, 0, bias=True))

        self.bbox_conv = StackedConvBlocks(
                    1, nn.Conv3d, bbox_conv_input_features, num_queries,
                    1, 1, 0, norm_op, norm_op_kwargs,
                    dropout_op, dropout_op_kwargs, nn.Sigmoid, None, False)

        self.residual_classifier = StackedConvBlocks(
                    1, nn.Conv3d, bbox_conv_input_features, num_queries,
                    3, 1, 1, norm_op, norm_op_kwargs,
                    dropout_op, dropout_op_kwargs, nn.GELU, None, False)


        self.image_decoder_stages = nn.ModuleList(image_decoder_stages)
        # self.image_decoder_stages_first = nn.ModuleList(image_decoder_stages_first)
        self.image_decoder_transpconvs = nn.ModuleList(image_decoder_transpconvs)
        self.image_decoder_seg_layers = nn.ModuleList(image_decoder_seg_layers)
        self.image_decoder_skips = nn.ModuleList(image_decoder_skips)
        # self.image_decoder_skips_first = nn.ModuleList(image_decoder_skips_first)

        self.final_conv = nn.Sequential(
            StackedConvBlocks(1, conv_op, num_queries, 64, (3, 3, 3), 1, conv_bias, norm_op, norm_op_kwargs,
                dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first),
            nn.ConvTranspose3d(64, 64, (1,2,2), (1,2,2), bias=conv_bias,),
            torch.nn.InstanceNorm3d(64),
            nn.GELU(),
            nn.ConvTranspose3d(64, 64, (1,2,2), (1,2,2), bias=conv_bias,),
            torch.nn.InstanceNorm3d(64),
            nn.GELU(),
            nn.Conv3d(64, num_queries, (3, 3, 3), 1, 1, bias=conv_bias,)
        )

    def _get_bbox(self, mask: torch.Tensor) -> torch.Tensor:

        # MASK_B, MASK_C, MASK_H, MASK_W = mask.shape

        # mask = rearrange(mask, 'b c h w -> (b c) h w')
        bbox = self.masks_to_boxes(mask)

        # bbox = rearrange(bbox, '(b c) n -> b c n', c=MASK_C)

        return bbox

    
    def _one_hot(self, gt):
        max_value = torch.amax(gt)

        shp_x = [*gt.shape]
        shp_x[1] = int(max_value)+1

        gt = gt.long()
        y_onehot = torch.zeros(shp_x, device=gt.device)
        y_onehot.scatter_(1, gt, 1)

        return y_onehot
    
    def masks_to_boxes(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Compute the bounding boxes around the provided masks.

        Returns a [N, 4] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
        ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

        Args:
            masks (Tensor[N, H, W]): masks to transform where N is the number of masks
                and (H, W) are the spatial dimensions.

        Returns:
            Tensor[N, 4]: bounding boxes
        """

        if masks.numel() == 0:
            return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

        n, nq, h, w = masks.shape

        bounding_boxes = torch.zeros((n, nq, 2, 2), device=masks.device, dtype=torch.float)

        for index, mask in enumerate(masks):
            
            for index_nq, nq_mask in enumerate(mask):
                if 1 in nq_mask:
                    y, x = torch.where(nq_mask != 0)
                    bounding_boxes[index, index_nq, 0, 0] = torch.min(x) / w
                    bounding_boxes[index, index_nq, 0, 1] = torch.min(y) / h
                    bounding_boxes[index, index_nq, 1, 0] = torch.max(x) / w
                    bounding_boxes[index, index_nq, 1, 1] = torch.max(y) / h

        return bounding_boxes
    
    def lxlypxpy_x1y1x2y2(self, bbox):
        return torch.cat([bbox[:,:, 0:1, :], (1-bbox[:,:, 0:1, :]) * bbox[:,:, 1:2, :] + bbox[:,:, 0:1, :]], dim=2)
    
    def forward(self, input_x, second_stage=False):  #, input_mask):

        INPUT_B, INPUT_C, INPUT_T, INPUT_H, INPUT_W = input_x.shape

        image = self.first_conv(input_x)

        # Get predictioin mask
        # image = rearrange(image, '(b t) c h w -> b c t h w', t=INPUT_T)
        image_embeddings = self.sam_model.image_encoder(image)  # (B,256,64,64)
        
        seg_outputs = []
        bbox_outputs = []
        classifier_outputs = []

        for s in range(len(self.image_decoder_stages)):
            # print(s, len(image_embeddings)-s-1, len(image_embeddings))
            image_embeddings_decoder_vit = rearrange(image_embeddings[len(image_embeddings)-s-1], '(b t) c h w -> b c t h w', t=INPUT_T)

            if s == len(self.image_decoder_stages) - 1:
                image_decoder_x = self.image_decoder_transpconvs[s-1](image_embeddings_decoder_below)
                image_embeddings_decoder_vit = self.image_decoder_skips[s-1](image_embeddings_decoder_vit)
                # print(s, image_decoder_x.shape, image_embeddings_decoder_vit.shape, self.image_decoder_skips[s-1])
                image_decoder_x = torch.concat([image_decoder_x, image_embeddings_decoder_vit], dim=1)
                image_decoder_x = self.image_decoder_stages[s](image_decoder_x)
                
            elif s == 0:
                image_decoder_x = self.image_decoder_stages[s](image_embeddings_decoder_vit)
            else:
                image_decoder_x = self.image_decoder_transpconvs[s-1](image_embeddings_decoder_below)
                image_embeddings_decoder_vit = self.image_decoder_skips[s-1](image_embeddings_decoder_vit)
                # print(s, image_decoder_x.shape, image_embeddings_decoder_vit.shape)
                image_decoder_x = torch.concat([image_decoder_x, image_embeddings_decoder_vit], dim=1)
                image_decoder_x = self.image_decoder_stages[s](image_decoder_x)
            
            bbox_outputs.append(torch.nn.functional.adaptive_avg_pool3d(image_decoder_x, output_size=(image_decoder_x.shape[-3], 2, 2)))
            classifier_outputs.append(torch.nn.functional.adaptive_avg_pool3d(image_decoder_x, output_size=(image_decoder_x.shape[-3], 16, 16)))
            
            if s == len(self.image_decoder_stages)-1:
                seg_outputs.append(self.image_decoder_seg_layers[0](image_decoder_x))

            image_embeddings_decoder_below = image_decoder_x

        input_mask = torch.sigmoid(seg_outputs[0])

        MASK_B, MASK_NQ, MASK_T, MASK_H, MASK_W = input_mask.shape 
        input_mask = rearrange(input_mask, 'b nq t h w -> (b t) nq h w')
        out= {'pred_aux_masks': input_mask}

        # input_mask = rearrange(input_mask, 'b nq h w -> (b nq) h w')

        input_mask = input_mask>0.5
        input_mask = input_mask.to(input_x.dtype)

        bbox_from_masks = self.masks_to_boxes(input_mask)

        bbox_outputs = torch.concat(bbox_outputs, dim=1)
        bbox_outputs = self.bbox_conv(bbox_outputs)
        bbox_outputs = rearrange(bbox_outputs, 'b nq t h w -> (b t) nq h w')

        bbox_outputs = self.lxlypxpy_x1y1x2y2(bbox_outputs)

        bbox_outputs = (bbox_outputs + bbox_from_masks) / 2

        out['pred_aux_bboxes'] = rearrange(bbox_outputs, 'b n h w -> b n (h w)')

        bbox = torch.cat([bbox_outputs[..., 0:1] * INPUT_W, bbox_outputs[..., 1:2] * INPUT_H], dim=-1)
        bbox = torch.floor(bbox)

        BBOX_BS, BBOX_N, _, _ = bbox.shape
        bbox = rearrange(bbox, 'b n h w -> (b n) (h w)')
        
        # bbox = self.sam_trans.apply_boxes(bbox, (INPUT_H, INPUT_W))
        box_tensor = torch.as_tensor(bbox, dtype=input_x.dtype, device=self.device)


        #### residual classifier #####
        # (B M T 1 1) -> (B M T)
        classifier_outputs = torch.concat(classifier_outputs, dim=1)
        # classifier_outputs = rearrange(classifier_outputs, 'b m t -> (b t) m')
        classifier_outputs = self.residual_classifier(classifier_outputs)
        classifier_outputs = rearrange(classifier_outputs, 'b nq t h w -> (b t nq) (h w)')
        classifier_outputs = classifier_outputs[:, None, :]

        # mask_input (np.ndarray): A low resolution mask input to the model, typically
        #     coming from a previous prediction iteration. Has form Bx1xHxW, where
        #     for SAM, H=W=256. Masks returned by a previous iteration of the
        #     predict method do not need further transformation.
        input_mask = rearrange(input_mask, 'b nq h w -> (b nq) h w')
        
        sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
            points=None,
            boxes=box_tensor,
            masks=TF.resize(torch.unsqueeze(input_mask, 1), (INPUT_H//4, INPUT_W//4), antialias=True, interpolation=torchvision.transforms.InterpolationMode.NEAREST),
        )

        # Predict masks
        low_res_masks, _, output_classes = self.sam_model.mask_decoder(
            image_embeddings=image_embeddings[-1].to(
                self.device
            ),  # (B, 256, 64, 64)
            image_pe=self.sam_model.prompt_encoder.get_dense_pe(image_embeddings[-1].shape[0], size=(64, 64)),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
            classifier_outputs=classifier_outputs,
        )

        #(B, 1, H, W) -> (B, H, W)
        mask_predictions = torch.squeeze(low_res_masks)
        mask_predictions = rearrange(mask_predictions, '(b c) h w -> b c h w', c=MASK_NQ)
        mask_predictions = rearrange(mask_predictions, '(b t) c h w -> b c t h w', t=INPUT_T)

        mask_predictions = self.final_conv(mask_predictions)

        mask_predictions = rearrange(mask_predictions, 'b nq t h w -> (b t) nq h w')

        if not self.decoder.deep_supervision:

            #(BN, NUM_CLASEES+1) -> (B, N, NUM_CLASEES+1)
            output_classes = rearrange(output_classes, '(b n) c -> b n c', n=MASK_NQ)

            out['pred_logits'] = output_classes
            out['pred_masks'] = mask_predictions#.tanh()

            output_classes = F.softmax(output_classes, dim=-1)[..., :-1]
            mask_predictions = mask_predictions.sigmoid()
            semseg = torch.einsum("nqc,nqhw->nchw", output_classes, mask_predictions)
            semseg = rearrange(semseg, '(b t) c h w -> b c t h w', t=INPUT_T)

            return semseg
        else:

            #(BN, NUM_CLASEES+1) -> (B, N, NUM_CLASEES+1)
            output_classes = rearrange(output_classes, '(b n) c -> b n c', n=MASK_NQ)

            out['pred_logits'] = output_classes
            out['pred_masks'] = mask_predictions#.tanh()

            return out
        
