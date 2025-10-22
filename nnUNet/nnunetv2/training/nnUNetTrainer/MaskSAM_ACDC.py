import inspect
import multiprocessing
import os
import shutil
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from time import time, sleep
from typing import Union, Tuple, List
from nnunetv2.training.loss.criterion import SetCriterion, SetCriterion_RemoveSamplePoints
from nnunetv2.training.loss.matcher import HungarianMatcher, HungarianMatcherAndAux, HungarianMatcherAndAux_2_4, HungarianMatcherAndAux_RemoveSamplePoints
from einops import rearrange
from nnunetv2.utilities.cc_torch import connected_components_labeling
from torch.nn import functional as F

import numpy as np
import torch
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from torch._dynamo import OptimizedModule

from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.training.data_augmentation.custom_transforms.cascade_transforms import MoveSegAsOneHotToData, \
    ApplyRandomBinaryOperatorTransform, RemoveRandomConnectedComponentFromOneHotEncodingTransform
from nnunetv2.training.data_augmentation.custom_transforms.deep_supervision_donwsampling import \
    DownsampleSegForDSTransform2
from nnunetv2.training.data_augmentation.custom_transforms.limited_length_multithreaded_augmenter import \
    LimitedLenWrapper
from nnunetv2.training.data_augmentation.custom_transforms.masking import MaskTransform
from nnunetv2.training.data_augmentation.custom_transforms.region_based_training import \
    ConvertSegmentationToRegionsTransform
from nnunetv2.training.data_augmentation.custom_transforms.transforms_for_dummy_2d import Convert2DTo3DTransform, \
    Convert3DTo2DTransform
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.dataloading.utils import get_case_identifiers, unpack_dataset
from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from sklearn.model_selection import KFold
from torch import autocast, nn
from torch import distributed as dist
from torch.cuda import device_count
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.sam.sam_model_2024_acdc_patch1024_tqreshape import SAMAdapter_2024_ACDC_Patch1024_TQReshape
from nnunetv2.training.data_augmentation.custom_transforms.instances import Instances
from torch import nn

class MaskSAM_ACDC(nnUNetTrainer):
    """
    Info for Fabian: same as internal nnUNetTrainerV2Adapter1
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 500
        self.initial_lr = 1e-3

    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision,
                frames=self.configuration_manager.patch_size[0],
            ).to(self.device)
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            self.latest_loss = self._build_latest_loss()

            self.match_loss = self._match_loss()

            # torch 2.2.2 crashes upon compiling CE loss
            # if self._do_i_compile():
            #     self.loss = torch.compile(self.loss)
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True,
                                   allow_init: bool = True,
                                   frames: int=8) -> nn.Module:
        """
        This is where you build the architecture according to the plans. There is no obligation to use
        get_network_from_plans, this is just a utility we use for the nnU-Net default architectures. You can do what
        you want. Even ignore the plans and just return something static (as long as it can process the requested
        patch size)
        but don't bug us with your bugs arising from fiddling with this :-P
        This is the function that is called in inference as well! This is needed so that all network architecture
        variants can be loaded at inference time (inference will use the same nnUNetTrainer that was used for
        training, so if you change the network architecture during training by deriving a new trainer class then
        inference will know about it).

        If you need to know how many segmentation outputs your custom architecture needs to have, use the following snippet:
        > label_manager = plans_manager.get_label_manager(dataset_json)
        > label_manager.num_segmentation_heads
        (why so complicated? -> We can have either classical training (classes) or regions. If we have regions,
        the number of outputs is != the number of classes. Also there is the ignore label for which no output
        should be generated. label_manager takes care of all that for you.)

        """

        network = SAMAdapter_2024_ACDC_Patch1024_TQReshape(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            frames=frames,
            **arch_init_kwargs
        )

        return network
    
    def _get_deep_supervision_scales(self):
        deep_supervision_scales = list(list(i) for i in 1 / np.cumprod(np.vstack(
            self.configuration_manager.pool_op_kernel_sizes), axis=0))[:-1]
        return deep_supervision_scales

    def _match_loss(self):
        # loss weights
        class_weight = 0.20
        dice_weight = 1.5
        mask_weight = 1.5
        aux_bbox_weight = 0.50
        aux_mask_weight = 0.50
        aux_giou_weight = 0.20
        num_points = 128*128*2 #32768 #12544*4 #65536

        # building criterion
        matcher = HungarianMatcherAndAux_2_4(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=num_points,
            cost_aux_bbox=aux_bbox_weight,
            cost_aux_mask=aux_mask_weight,
            cost_aux_giou=aux_giou_weight,
        )

        losses = ["labels", "masks", "aux_masks", "aux_bboxes"]
        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, 
                        "loss_dice": dice_weight, "loss_aux_mask": aux_mask_weight,
                        "loss_aux_dice": aux_mask_weight, "loss_aux_bbox": aux_bbox_weight, 
                        "loss_aux_giou": aux_giou_weight}
        
        # print(self.label_manager.num_segmentation_heads)
        criterion = SetCriterion(
            self.label_manager.num_segmentation_heads,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=0.1,
            losses=losses,
            num_points=num_points,
            oversample_ratio=2.0,
            importance_sample_ratio=0.75,
        ).to(self.device)


        return criterion

    def _build_loss(self):
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        # weights[-1] = 0
        weights[5:] = 0

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def _build_latest_loss(self):
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        # weights[-1] = 0
        weights[1:] = 0

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss


    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            case_identifiers = get_case_identifiers(self.preprocessed_dataset_folder)
            tr_keys = case_identifiers
            val_keys = tr_keys
        else:
            splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
            dataset = nnUNetDataset(self.preprocessed_dataset_folder, case_identifiers=None,
                                    num_images_properties_loading_threshold=0,
                                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                splits = []
                all_keys_sorted = np.sort(list(dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append({})
                    splits[-1]['train'] = list(train_keys)
                    splits[-1]['val'] = list(test_keys)
                save_json(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_json(splits_file)
                self.print_to_log_file("The split file contains %d splits." % len(splits))

            splits[self.fold]['train']=['patient001_frame01', 'patient001_frame12', 'patient004_frame01',
       'patient004_frame15', 'patient005_frame01', 'patient005_frame13',
       'patient006_frame01', 'patient006_frame16', 'patient007_frame01',
       'patient007_frame07', 'patient010_frame01', 'patient010_frame13',
       'patient011_frame01', 'patient011_frame08', 'patient013_frame01',
       'patient013_frame14', 'patient015_frame01', 'patient015_frame10',
       'patient016_frame01', 'patient016_frame12', 'patient018_frame01',
       'patient018_frame10', 'patient019_frame01', 'patient019_frame11',
       'patient020_frame01', 'patient020_frame11', 'patient021_frame01',
       'patient021_frame13', 'patient022_frame01', 'patient022_frame11',
       'patient023_frame01', 'patient023_frame09', 'patient025_frame01',
       'patient025_frame09', 'patient026_frame01', 'patient026_frame12',
       'patient027_frame01', 'patient027_frame11', 'patient028_frame01',
       'patient028_frame09', 'patient029_frame01', 'patient029_frame12',
       'patient030_frame01', 'patient030_frame12', 'patient031_frame01',
       'patient031_frame10', 'patient032_frame01', 'patient032_frame12',
       'patient033_frame01', 'patient033_frame14', 'patient034_frame01',
       'patient034_frame16', 'patient035_frame01', 'patient035_frame11',
       'patient036_frame01', 'patient036_frame12', 'patient037_frame01',
       'patient037_frame12', 'patient038_frame01', 'patient038_frame11',
       'patient039_frame01', 'patient039_frame10', 'patient040_frame01',
       'patient040_frame13', 'patient041_frame01', 'patient041_frame11',
       'patient043_frame01', 'patient043_frame07', 'patient044_frame01',
       'patient044_frame11', 'patient045_frame01', 'patient045_frame13',
       'patient046_frame01', 'patient046_frame10', 'patient047_frame01',
       'patient047_frame09', 'patient050_frame01', 'patient050_frame12',
       'patient051_frame01', 'patient051_frame11', 'patient052_frame01',
       'patient052_frame09', 'patient054_frame01', 'patient054_frame12',
       'patient056_frame01', 'patient056_frame12', 'patient057_frame01',
       'patient057_frame09', 'patient058_frame01', 'patient058_frame14',
       'patient059_frame01', 'patient059_frame09', 'patient060_frame01',
       'patient060_frame14', 'patient061_frame01', 'patient061_frame10',
       'patient062_frame01', 'patient062_frame09', 'patient063_frame01',
       'patient063_frame16', 'patient065_frame01', 'patient065_frame14',
       'patient066_frame01', 'patient066_frame11', 'patient068_frame01',
       'patient068_frame12', 'patient069_frame01', 'patient069_frame12',
       'patient070_frame01', 'patient070_frame10', 'patient071_frame01',
       'patient071_frame09', 'patient072_frame01', 'patient072_frame11',
       'patient073_frame01', 'patient073_frame10', 'patient074_frame01',
       'patient074_frame12', 'patient075_frame01', 'patient075_frame06',
       'patient076_frame01', 'patient076_frame12', 'patient077_frame01',
       'patient077_frame09', 'patient078_frame01', 'patient078_frame09',
       'patient080_frame01', 'patient080_frame10', 'patient082_frame01',
       'patient082_frame07', 'patient083_frame01', 'patient083_frame08',
       'patient084_frame01', 'patient084_frame10', 'patient085_frame01',
       'patient085_frame09', 'patient086_frame01', 'patient086_frame08',
       'patient087_frame01', 'patient087_frame10']
            splits[self.fold]['val']=['patient089_frame01', 'patient089_frame10', 'patient090_frame04',
       'patient090_frame11', 'patient091_frame01', 'patient091_frame09',
       'patient093_frame01', 'patient093_frame14', 'patient094_frame01',
       'patient094_frame07', 'patient096_frame01', 'patient096_frame08',
       'patient097_frame01', 'patient097_frame11', 'patient098_frame01',
       'patient098_frame09', 'patient099_frame01', 'patient099_frame09',
       'patient100_frame01', 'patient100_frame13']

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            if any([i in val_keys for i in tr_keys]):
                self.print_to_log_file('WARNING: Some validation cases are also in the training set. Please check the '
                                       'splits.json or ignore if this is intentional.')
        return tr_keys, val_keys

    def on_train_start(self):
        if not self.was_initialized:
            self.initialize()

        maybe_mkdir_p(self.output_folder)

        # make sure deep supervision is on in the network
        self.set_deep_supervision_enabled(True)

        self.print_plans()
        empty_cache(self.device)

        # maybe unpack
        if self.unpack_dataset and self.local_rank == 0:
            self.print_to_log_file('unpacking dataset...')
            unpack_dataset(self.preprocessed_dataset_folder, unpack_segmentation=True, overwrite_existing=False,
                           num_processes=max(1, round(get_allowed_n_proc_DA() // 2)))
            self.print_to_log_file('unpacking done...')

        if self.is_ddp:
            dist.barrier()

        # dataloaders must be instantiated here because they need access to the training data which may not be present
        # when doing inference
        self.dataloader_train, self.dataloader_val = self.get_dataloaders()

        # copy plans and dataset.json so that they can be used for restoring everything we need for inference
        save_json(self.plans_manager.plans, join(self.output_folder_base, 'plans.json'), sort_keys=False)
        save_json(self.dataset_json, join(self.output_folder_base, 'dataset.json'), sort_keys=False)

        # we don't really need the fingerprint but its still handy to have it with the others
        shutil.copy(join(self.preprocessed_dataset_folder_base, 'dataset_fingerprint.json'),
                    join(self.output_folder_base, 'dataset_fingerprint.json'))

        # produces a pdf in output folder
        self.plot_network_architecture()

        self._save_debug_information()

    def on_train_end(self):
        self.save_checkpoint(join(self.output_folder, "checkpoint_final.pth"))
        # now we can delete latest
        if self.local_rank == 0 and isfile(join(self.output_folder, "checkpoint_latest.pth")):
            os.remove(join(self.output_folder, "checkpoint_latest.pth"))

        # shut down dataloaders
        old_stdout = sys.stdout
        with open(os.devnull, 'w') as f:
            sys.stdout = f
            if self.dataloader_train is not None:
                self.dataloader_train._finish()
            if self.dataloader_val is not None:
                self.dataloader_val._finish()
            sys.stdout = old_stdout

        empty_cache(self.device)
        self.print_to_log_file("Training done.")

    def on_train_epoch_start(self):
        self.network.train()
        self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)


    def masks_to_boxes(self, masks):
        """Compute the bounding boxes around the provided masks
        The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.
        Returns a [N, 4] tensors, with the boxes in xyxy format
        """
        if masks.numel() == 0:
            return torch.zeros((0, 4), device=masks.device)

        h, w = masks.shape[-2:]

        y = torch.arange(0, h, dtype=torch.float, device=masks.device)
        x = torch.arange(0, w, dtype=torch.float, device=masks.device)
        y, x = torch.meshgrid(y, x)

        x_mask = (masks * x.unsqueeze(0))
        x_max = x_mask.flatten(1).max(-1)[0]
        x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

        y_mask = (masks * y.unsqueeze(0))
        y_max = y_mask.flatten(1).max(-1)[0]
        y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

        return torch.stack([x_min/w, y_min/h, x_max/w, y_max/h], 1)

    def semseg_insanceseg(self, semseg):
        patch_size = self.configuration_manager.patch_size
        bs, _, t, h, w = semseg.shape

        semseg = rearrange(semseg, 'b c d h w -> (b d) c h w')

        bsfs, _, _, _ = semseg.shape
        data_dict = {}

        skip = False

        for b in range(bsfs):
            instances = Instances((h, w))

            seg = semseg[b,0,...]
            
            classes = []
            masks = []
            ids = torch.unique(seg).to(torch.long)

            for id in ids:
                if id == 0:
                    mask=torch.where(seg==id, 1, 0)
                    masks.append(mask)
                    classes.append(id)
                else:
                    mask=torch.where(seg==id, 1, 0)
                    masks.append(mask)
                    classes.append(id) 

            # print(len(classes))
            if len(classes)>48:
                print(len(classes))
                skip = True

            instances.gt_classes = torch.stack(classes) #dtype=torch.int64)
            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, h, w))
                instances.gt_bboxes = torch.zeros((0, 4))
            else:
                # masks = BitMasks(
                #     torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                # )
                masks = torch.stack(masks)
                instances.gt_masks = masks
                # import datetime
                # print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %f"))
                # gt_bboxes1 = self.masks_to_boxes(instances.gt_masks)
                # print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %f"))
                instances.gt_bboxes = self.masks_to_boxes(instances.gt_masks)
                # gt_bboxes2 = self.my_masks_to_boxes(instances.gt_masks)
                # print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %f"))

            data_dict[b] = {'instances':instances}

        return data_dict, skip
    
    def semseg_insanceseg_targets(self, semseg):
        # patch_size = self.configuration_manager.patch_size
        _, _, _, h, w = semseg.shape
        new_targets = []

        semseg = rearrange(semseg, 'b c d h w -> (b d) c h w')

        bsfs, _, _, _ = semseg.shape
        # data_dict = {}
        for b in range(bsfs):
            # instances = Instances((h, w))

            seg = semseg[b,0,...]
            
            classes = []
            masks = []
            ids = torch.unique(seg).to(torch.long)

            for id in ids:
                mask=torch.where(seg==id, 1, 0)

                cc_out = connected_components_labeling(mask.to(torch.uint8))
                
                case_ids = torch.unique(cc_out)
                
                if case_ids.shape[0] > 2:
                    for case_id in case_ids[1:]:
                        case_mask=torch.where(cc_out==case_id, 1, 0)
                        masks.append(case_mask)
                        classes.append(id)
                else:
                    masks.append(mask)
                    classes.append(id) 

            new_gt_classes = torch.stack(classes) #dtype=torch.int64)
            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                new_gt_masks = torch.zeros((0, h, w))
                new_gt_bboxes = torch.zeros((0, 4))
            else:
                # masks = BitMasks(
                #     torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                # )
                masks = torch.stack(masks)
                new_gt_masks = masks
                # import datetime
                # print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %f"))
                # gt_bboxes1 = self.masks_to_boxes(instances.gt_masks)
                # print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %f"))
                new_gt_bboxes = self.masks_to_boxes(new_gt_masks)
                # gt_bboxes2 = self.my_masks_to_boxes(instances.gt_masks)
                # print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %f"))

            # data_dict[b] = {'instances':instances}
            new_targets.append(
                {
                    "labels": new_gt_classes,
                    "masks": new_gt_masks,
                    "bboxes": new_gt_bboxes
                }
            )

        return new_targets

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            
            # bboxes = self.masks_to_normalized_boxes(padded_masks, (h_pad, w_pad))

            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                    "bboxes": targets_per_image.gt_bboxes
                }
            )
        return new_targets


    def train_step(self, batch: dict, second_stage, dataloader_train) -> dict:
        data = batch['data']
        targets = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(targets, list):
            # target = [i.to(self.device, non_blocking=True) for i in target]
            targets = targets[0].to(self.device, non_blocking=True)
        else:
            targets = targets.to(self.device, non_blocking=True)

        batch_instances, skip = self.semseg_insanceseg(targets)# mask classification target

        while skip:
            batch = next(dataloader_train)

            data = batch['data']
            targets = batch['target']

            data = data.to(self.device, non_blocking=True)
            if isinstance(targets, list):
                # target = [i.to(self.device, non_blocking=True) for i in target]
                targets = targets[0].to(self.device, non_blocking=True)
            else:
                targets = targets.to(self.device, non_blocking=True)

            batch_instances, skip = self.semseg_insanceseg(targets)# mask classification target

        if "instances" in batch_instances[0]:
            gt_instances = [batch_instances[x]["instances"].to(self.device) for x in batch_instances]
            # gt_bboxes = [batch_instances[x]["bboxes"].to(self.device) for x in batch_instances]
            targets = self.prepare_targets(gt_instances, data)
        else:
            targets = None

        batch_instances = None
        gt_instances = None
        # targets = self.semseg_insanceseg_targets(target)# mask classification target

        self.optimizer.zero_grad()
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():

            outputs = self.network(data, second_stage=second_stage)
            
            del data

            # bipartite matching-based loss
            losses = self.match_loss(outputs, targets)

            l = 0

            for k in list(losses.keys()):
                if k in self.match_loss.weight_dict:
                    l += self.match_loss.weight_dict[k] * losses[k]
                    # print(k, l, self.match_loss.weight_dict[k] * losses[k])
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)

        # torch.autograd.set_detect_anomaly(True)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}

    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs['loss'])
            loss_here = np.vstack(losses_tr).mean()
        else:
            loss_here = np.mean(outputs['loss'])

        self.logger.log('train_losses', loss_here, self.current_epoch)

    def on_validation_epoch_start(self):
        self.network.eval()

    def validation_step(self, batch: dict, second_stage: bool=False) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = target[0].to(self.device, non_blocking=True)
        else:
            target = target.to(self.device, non_blocking=True)

        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():

            out = self.network(data, second_stage=second_stage)

            output_classes = out['pred_logits'] 
            mask_predictions = out['pred_masks']

            output_classes = F.softmax(output_classes, dim=-1)[..., :-1]
            mask_predictions = mask_predictions.sigmoid()
            semseg = torch.einsum("nqc,nqhw->nchw", output_classes, mask_predictions)
            semseg = rearrange(semseg, '(b t) c h w -> b c t h w', t=self.configuration_manager.patch_size[0])
            
            l = self.latest_loss([semseg,], [target,])
            output = semseg


        # we only need the output with the highest output resolution
        # output = output[0]
        # target = target[0]

        # output = mask_predictions_r[0]
        target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, len(output.shape)))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        if self.is_ddp:
            world_size = dist.get_world_size()

            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)

            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)

            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)

            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(outputs_collated['loss'])

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
                                           zip(tp, fp, fn)]]
        mean_fg_dice = np.nanmean(global_dc_per_class)
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)

    def on_epoch_start(self):
        self.logger.log('epoch_start_timestamps', time(), self.current_epoch)
        empty_cache(self.device)

    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        # todo find a solution for this stupid shit
        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                               self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1


    def run_training(self):
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train), epoch > self.second_stage, self.dataloader_train))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val), epoch > self.second_stage))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

        self.on_train_end()
