export nnUNet_raw="/data/xiebin/nnunet/DATASETS/nnUNet_raw"
export nnUNet_preprocessed="/data/xiebin/nnunet/DATASETS/nnUNet_preprocessed"
export nnUNet_results="/data/xiebin/nnunet/DATASETS/nnUNet_results"


CUDA_VISIBLE_DEVICES=0,1 nnUNetv2_train 52 3d_fullres 0 -tr MaskSAM_AMOS -p nnUNetPlans_9_512_512_b2 -num_gpus 2 --c

CUDA_VISIBLE_DEVICES=0,1 nnUNetv2_train 52 3d_fullres 1 -tr MaskSAM_AMOS -p nnUNetPlans_9_512_512_b2 -num_gpus 2 --c

CUDA_VISIBLE_DEVICES=0,1 nnUNetv2_train 52 3d_fullres 2 -tr MaskSAM_AMOS -p nnUNetPlans_9_512_512_b2 -num_gpus 2 --c

CUDA_VISIBLE_DEVICES=0,1 nnUNetv2_train 52 3d_fullres 3 -tr MaskSAM_AMOS -p nnUNetPlans_9_512_512_b2 -num_gpus 2 --c

CUDA_VISIBLE_DEVICES=0,1 nnUNetv2_train 52 3d_fullres 4 -tr MaskSAM_AMOS -p nnUNetPlans_9_512_512_b2 -num_gpus 2 --c
