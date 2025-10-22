# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial

from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer, PromptEncoderTPosEmbed, MaskDecoderClassifier, ImageEncoderViTGlobalOutputAdapter3DTPosEmbedTFusionMiddleDWConv, TwoWayTransformerAdapter3DTFusionMiddleDWConvACDCTQReshape


def build_sam_vit_h(checkpoint=None, num_frames: int=4,):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
        num_frames=num_frames,
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(checkpoint=None, num_frames: int=4,):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
        num_frames=num_frames,
    )


def build_sam_vit_b(checkpoint=None, num_frames: int=4,):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        num_frames=num_frames,
    )


def build_sam_vit_b_adapter_2024_acdc_patch1024_tqreshape(checkpoint=None, num_frames: int=4, num_classes: int=14, num_queries: int=8):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        num_frames=num_frames,
        num_classes=num_classes,
        num_queries=num_queries,
        adapter="adapter_2024_acdc_patch1024_tqreshape",
    )

def build_sam_vit_h_adapter_2024_acdc_smallpatch_tqreshape(checkpoint=None, num_frames: int=4, num_classes: int=14, num_queries: int=8):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
        num_frames=num_frames,
        num_classes=num_classes,
        num_queries=num_queries,
        adapter="adapter_2024_acdc_smallpatch_tqreshape",
    )

def build_sam_vit_h_adapter_2024_amos_patch512_tqreshape(checkpoint=None, num_frames: int=4, num_classes: int=14, num_queries: int=8):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
        num_frames=num_frames,
        num_classes=num_classes,
        num_queries=num_queries,
        adapter="adapter_2024_amos_patch512_tqreshape",
    )


sam_model_registry = {
    "vit_b_adapter_2024_acdc_patch1024_tqreshape": build_sam_vit_b_adapter_2024_acdc_patch1024_tqreshape,
    "vit_h_adapter_2024_acdc_smallpatch_tqreshape": build_sam_vit_h_adapter_2024_acdc_smallpatch_tqreshape,
    "vit_h_adapter_2024_amos_patch512_tqreshape": build_sam_vit_h_adapter_2024_amos_patch512_tqreshape,
    }


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
    num_frames=4,
    num_classes=14, 
    num_queries=14,
    adapter="",
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    
    if adapter == "adapter_2024_acdc_patch1024_tqreshape":
        sam = Sam(
            image_encoder=ImageEncoderViTGlobalOutputAdapter3DTPosEmbedTFusionMiddleDWConv(
                depth=encoder_depth,
                embed_dim=encoder_embed_dim,
                img_size=image_size,
                mlp_ratio=4,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                num_heads=encoder_num_heads,
                patch_size=vit_patch_size,
                qkv_bias=True,
                use_rel_pos=True,
                global_attn_indexes=encoder_global_attn_indexes,
                window_size=14,
                out_chans=prompt_embed_dim,
                num_frames=num_frames,
            ),
            prompt_encoder=PromptEncoderTPosEmbed(
                embed_dim=prompt_embed_dim,
                image_embedding_size=(image_embedding_size//1, image_embedding_size//1),
                input_image_size=(image_size//1, image_size//1),
                mask_in_chans=16,
                num_frames=num_frames,
            ),
            mask_decoder=MaskDecoderClassifier(
                num_multimask_outputs=3,
                transformer=TwoWayTransformerAdapter3DTFusionMiddleDWConvACDCTQReshape(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                    num_frames=num_frames,
                    num_queries=num_queries,
                    adapter_feature_size = (64,64),
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
                num_classes=num_classes,
            ),
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
        )
    elif adapter == "adapter_2024_acdc_smallpatch_tqreshape":
        sam = Sam(
            image_encoder=ImageEncoderViTGlobalOutputAdapter3DTPosEmbedTFusionMiddleDWConv(
                depth=encoder_depth,
                embed_dim=encoder_embed_dim,
                img_size=image_size,
                mlp_ratio=4,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                num_heads=encoder_num_heads,
                patch_size=vit_patch_size,
                qkv_bias=True,
                use_rel_pos=True,
                global_attn_indexes=encoder_global_attn_indexes,
                window_size=14,
                out_chans=prompt_embed_dim,
                num_frames=num_frames,
            ),
            prompt_encoder=PromptEncoderTPosEmbed(
                embed_dim=prompt_embed_dim,
                image_embedding_size=(image_embedding_size//4, image_embedding_size//4),
                input_image_size=(image_size//1, image_size//1),
                mask_in_chans=16,
                num_frames=num_frames,
            ),
            mask_decoder=MaskDecoderClassifier(
                num_multimask_outputs=3,
                transformer=TwoWayTransformerAdapter3DTFusionMiddleDWConvACDCTQReshape(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                    num_frames=num_frames,
                    num_queries=num_queries,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
                num_classes=num_classes,
            ),
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
        )
    elif adapter == "adapter_2024_amos_patch512_tqreshape":
        sam = Sam(
            image_encoder=ImageEncoderViTGlobalOutputAdapter3DTPosEmbedTFusionMiddleDWConv(
                depth=encoder_depth,
                embed_dim=encoder_embed_dim,
                img_size=image_size,
                mlp_ratio=4,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                num_heads=encoder_num_heads,
                patch_size=vit_patch_size,
                qkv_bias=True,
                use_rel_pos=True,
                global_attn_indexes=encoder_global_attn_indexes,
                window_size=14,
                out_chans=prompt_embed_dim,
                num_frames=num_frames,
            ),
            prompt_encoder=PromptEncoderTPosEmbed(
                embed_dim=prompt_embed_dim,
                image_embedding_size=(image_embedding_size//2, image_embedding_size//2),
                input_image_size=(image_size//1, image_size//1),
                mask_in_chans=16,
                num_frames=num_frames,
            ),
            mask_decoder=MaskDecoderClassifier(
                num_multimask_outputs=3,
                transformer=TwoWayTransformerAdapter3DTFusionMiddleDWConvACDCTQReshape(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                    num_frames=num_frames,
                    num_queries=num_queries,
                    adapter_feature_size=(32, 32),
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
                num_classes=num_classes,
            ),
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
        )
    else:
        sam = Sam(
            image_encoder=ImageEncoderViT(
                depth=encoder_depth,
                embed_dim=encoder_embed_dim,
                img_size=image_size,
                mlp_ratio=4,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                num_heads=encoder_num_heads,
                patch_size=vit_patch_size,
                qkv_bias=True,
                use_rel_pos=True,
                global_attn_indexes=encoder_global_attn_indexes,
                window_size=14,
                out_chans=prompt_embed_dim,
                num_frames=num_frames,
            ),
            prompt_encoder=PromptEncoder(
                embed_dim=prompt_embed_dim,
                image_embedding_size=(image_embedding_size, image_embedding_size),
                input_image_size=(image_size, image_size),
               mask_in_chans=16,
                num_frames=num_frames,
            ),
            mask_decoder=MaskDecoder(
                num_multimask_outputs=3,
                transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                    num_frames=num_frames,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
            ),
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
        )
            
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict, False)
        # for name, param in sam.named_parameters():
        #     if 'temporal_embedding' not in name and 'Adapter' not in name:
        #         param.requires_grad = False
        #     else:
        #         if 'Adapter' in name and "mask_decoder" in name:
        #             param.requires_grad = False

        for name, param in sam.named_parameters():
            if 'temporal_embedding' not in name and 'Adapter' not in name and 'adapter' not in name:
                param.requires_grad = False
            # else:
            #     if 'Adapter' in name and "mask_decoder" in name:
            #         param.requires_grad = False
    return sam
