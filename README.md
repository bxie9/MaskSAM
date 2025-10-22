# MaskSAM: Auto-prompt SAM with Mask Classification for Volumetric Medical Image Segmentation

[[`Paper (ICCV2025)`](https://arxiv.org/abs/2403.14103)] 

![Overview of MaskSAM framework](assets/overview.png?raw=true "Overview of MaskSAM framework")

The **MaskSAM** is a novel mask classification prompt-free SAM adaptation framework for medical image segmentation. We design a prompt generator combined with the image encoder in SAM to generate a set of auxiliary classifier tokens, auxiliary binary masks, and auxiliary bounding boxes. Each pair of auxiliary mask and box prompts can solve the requirements of extra prompts. The semantic label prediction can be addressed by the sum of the auxiliary classifier tokens and the learnable global classifier tokens in the mask decoder of SAM. Meanwhile, we design a 3D depth-convolution adapter for image embeddings and a 3D depth-MLP adapter for prompt embeddings to efficiently fine-tune SAM. Our method achieves state-of-the-art performance on AMOS2022, 90.5% Dice.

## Installation

Install MaskSAM:

```
git clone https://github.com/bxie9/MaskSAM
cd MaskSAM; cd nnUNet
pip install -e .
```

## Training
#### 1. Dataset download
Datasets can be acquired via following links:

**Dataset I**
[ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/)

**Dataset II**
[The Synapse multi-organ CT dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)

**Dataset III**
[AMOS22](https://amos22.grand-challenge.org/)

#### 2. Setting up the datasets
After you have downloaded the datasets, you can follow the settings in [nnUNet](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md) for path configurations and preprocessing procedures. Finally, your folders should be organized as follows:

```
./nnUNet/
./DATASETS/
  ├── nnUNet_raw/
      ├── nnUNet_raw_data/
          ├── Dataset027_ACDC/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
          ├── Dataset028_Synapse/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
          ├── Dataset052_AMOS22_OnlyCT/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
  ├── nnUNet_preprocessed/
  ├── nnUNet_results/
```

After that, you can preprocess the above data using following commands:

```
nnUNetv2_plan_and_preprocess -d 27 -c 3d_fullres --verify_dataset_integrity

nnUNetv2_plan_and_preprocess -d 28 -c 3d_fullres --verify_dataset_integrity

nnUNetv2_plan_and_preprocess -d 52 -c 3d_fullres --verify_dataset_integrity
```

#### 3. Training
- Commands for training for different datasets:

```
bash bash/MaskSAM_ACDC.sh 

bash bash/MaskSAM_Synapse.sh 

bash bash/MaskSAM_AMOS.sh 
```

#### 4. Inference

- Commands for testing for different datasets:

```
CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -f 0 -tr MaskSAM_ACDC -c 3d_fullres -i /PATH/TO/INPUT/FOLD/ -o /PATH/TO/OUTPUT/FOLD/ -d 27

CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -f 0 -tr MaskSAM_Synapse -c 3d_fullres -i /PATH/TO/INPUT/FOLD/ -o /PATH/TO/OUTPUT/FOLD/ -d 28

CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -f 0 1 2 3 4 -tr MaskSAM_AMOS -c 3d_fullres -i /PATH/TO/INPUT/FOLD/ -o /PATH/TO/OUTPUT/FOLD/ -d 52

```

## Acknowledgments
Our code is based on [nnUNet](https://github.com/MIC-DKFZ/nnUNet) and [Segment Anything](https://github.com/facebookresearch/segment-anything). We appreciate the authors for their great works. 


## Citing MaskSAM

If you use SAM or SA-1B in your research, please use the following BibTeX entry.

```
@article{xie2024masksam,
  title={MaskSAM: Towards Auto-prompt SAM with Mask Classification for Volumetric Medical Image Segmentation},
  author={Xie, Bin and Tang, Hao and Duan, Bin and Cai, Dawen and Yan, Yan and Agam, Gady},
  journal={arXiv preprint arXiv:2403.14103},
  year={2024}
}
```