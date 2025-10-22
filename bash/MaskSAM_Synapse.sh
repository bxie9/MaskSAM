export nnUNet_raw="/data/xiebin/nnunet/DATASETS/nnUNet_raw"
export nnUNet_preprocessed="/data/xiebin/nnunet/DATASETS/nnUNet_preprocessed"
export nnUNet_results="/data/xiebin/nnunet/DATASETS/nnUNet_results"


CUDA_VISIBLE_DEVICES=0,1 nnUNetv2_train 52 3d_fullres 0 -tr MaskSAM_Synapse -p nnUNetPlans_26_256_256_b2 -num_gpus 2 --c
