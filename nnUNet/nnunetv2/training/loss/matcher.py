# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/matcher.py
# Modified by Jitesh Jain (https://github.com/praeclarumjj3)
# ------------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast
import numpy as np
from nnunetv2.training.loss import box_ops

def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output

def linear_sum_assignment_with_nan(cost_matrix):
    cost_matrix = np.asarray(cost_matrix)
    nan = np.isnan(cost_matrix).any()
    nan_all = np.isnan(cost_matrix).all()
    empty = cost_matrix.size == 0

    if not empty:
        if nan_all:
            print('Matrix contains all NaN values!')
        elif nan:
            print('Matrix contains NaN values!')

        if nan_all:
            cost_matrix = np.empty(shape=(0, 0))
        elif nan:
            cost_matrix[np.isnan(cost_matrix)] = 100

    # print("cost_matrix = ", cost_matrix[0][:3])
    return linear_sum_assignment(cost_matrix)

def w_func(grnd, w_type="square"):
        if w_type == "simple":
            return torch.reciprocal(grnd)
        if w_type == "square":
            return torch.reciprocal(grnd * grnd)
        return torch.ones_like(grnd)

def batch_generalized_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    smooth_nr = 1e-5
    smooth_dr = 1e-5

    reduce_axis: list[int] = torch.arange(1, len(inputs.shape)).tolist()

    inputs = inputs.to(torch.float64)
    targets = targets.to(torch.float64)

    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)

    intersection = torch.einsum("nc,mc->nm", inputs.to(torch.float32), targets.to(torch.float32))

    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]

    ground_o = torch.sum(targets, reduce_axis)
    # pred_o = torch.sum(inputs, reduce_axis)

    # denominator = ground_o + pred_o

    w = w_func(ground_o.float(), w_type = "simple")
    infs = torch.isinf(w)
    w[infs] = 0.0
    w = w + infs * torch.max(w)
    
    numer = 2.0 * (intersection * w) + smooth_nr
    denom = (denominator  * w) + smooth_dr
    return 1.0 - (numer / denom)

def batch_generalized_dice_loss_wow(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    smooth_nr = 1e-5
    smooth_dr = 1e-5

    reduce_axis: list[int] = torch.arange(1, len(inputs.shape)).tolist()

    inputs = inputs.to(torch.float64)
    targets = targets.to(torch.float64)

    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)

    intersection = torch.einsum("nc,mc->nm", inputs.to(torch.float32), targets.to(torch.float32))

    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]

    ground_o = torch.sum(targets, reduce_axis)
    # pred_o = torch.sum(inputs, reduce_axis)

    # denominator = ground_o + pred_o

    w = w_func(ground_o.float())
    infs = torch.isinf(w)
    w[infs] = 0.0
    w = w + infs * torch.max(w)
    
    numer = 2.0 * (intersection ) + smooth_nr
    denom = (denominator ) + smooth_dr
    return 1.0 - (numer / denom)

def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.to(torch.float64)
    targets = targets.to(torch.float64)
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs.to(torch.float32), targets.to(torch.float32))
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule

def batch_sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    inputs = inputs.to(torch.float64)
    targets = targets.to(torch.float64)
    hw = inputs.shape[1]

    prob = inputs.sigmoid()
    focal_pos = ((1 - prob) ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    focal_neg = (prob ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )
    if alpha >= 0:
        focal_pos = focal_pos * alpha
        focal_neg = focal_neg * (1 - alpha)

    loss = torch.einsum("nc,mc->nm", focal_pos, targets) + torch.einsum(
        "nc,mc->nm", focal_neg, (1 - targets)
    )

    return loss / hw


class HungarianMatcherAndAux_2_4(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, 
                    cost_dice: float = 1, num_points: int = 0, cost_aux_bbox: float = 1, cost_aux_mask: float = 1, cost_aux_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

        self.cost_aux_bbox = cost_aux_bbox
        self.cost_aux_mask = cost_aux_mask
        self.cost_aux_giou = cost_aux_giou
    
        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

        self.num_points = num_points

    @torch.no_grad()
    def detr_bbox(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                "pred_bboxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                        objects in the target) containing the class labels
                "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        # out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_bboxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        # tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["bboxes"] for v in targets])

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        # cost_giou = -box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(out_bbox), box_ops.box_cxcywh_to_xyxy(tgt_bbox))
        cost_giou = -box_ops.generalized_box_iou(out_bbox, tgt_bbox)

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["bboxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        indices = []

        # Iterate through batch size
        for b in range(bs):

            # We flatten to compute the cost matrices in a batch
            out_bbox = outputs["pred_aux_bboxes"][b]  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            tgt_bbox = targets[b]["bboxes"]

            # Compute the L1 cost between boxes
            cost_aux_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            cost_aux_giou = -box_ops.generalized_box_iou(out_bbox, tgt_bbox)

            out_aux_mask = outputs["pred_aux_masks"][b]  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_aux_mask)

            out_aux_mask = out_aux_mask[:, None]
            tgt_mask = tgt_mask[:, None]
            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_aux_mask.device)
            # get gt labels
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            out_aux_mask = point_sample(
                out_aux_mask,
                point_coords.repeat(out_aux_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            with torch.amp.autocast('cuda', enabled=False):
                out_aux_mask = out_aux_mask.float()
                tgt_mask = tgt_mask.float()
                # Compute the focal loss between masks
                cost_aux_mask = batch_sigmoid_ce_loss_jit(out_aux_mask, tgt_mask)
                # Compute the dice loss betwen masks
                cost_aux_dice = batch_dice_loss(out_aux_mask, tgt_mask)

            ########################################################################
            
            out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes]
            tgt_ids = targets[b]["labels"]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            # tgt_mask = targets[b]["masks"].to(out_mask)

            out_mask = out_mask[:, None]
            # tgt_mask = tgt_mask[:, None]
            # # all masks share the same set of points for efficient matching!
            # point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            # # get gt labels
            # tgt_mask = point_sample(
            #     tgt_mask,
            #     point_coords.repeat(tgt_mask.shape[0], 1, 1),
            #     align_corners=False,
            # ).squeeze(1)

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            with torch.amp.autocast('cuda', enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)
                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss(out_mask, tgt_mask)
            
            # Final cost matrix
            C = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
                + self.cost_aux_bbox * cost_aux_bbox
                + self.cost_aux_mask * cost_aux_mask
                + self.cost_aux_mask * cost_aux_dice
                + self.cost_aux_giou * cost_aux_giou
            )
            C = C.reshape(num_queries, -1).cpu()

            indices.append(linear_sum_assignment_with_nan(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        return self.memory_efficient_forward(outputs, targets) #, self.detr_bbox(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)

if __name__ == '__main__':
    
    pred = torch.rand((2,  128*128))
    ref = torch.randint(0, 2, (4,  128*128))


    dl_old = batch_generalized_dice_loss(pred, ref)
    dl_new = batch_generalized_dice_loss_wow(pred, ref)

    print(pred)
    print(ref)
    print(dl_old==dl_new)
    print(dl_new)
    print(dl_old)

    ref = torch.where(ref==1, 0, 0)
    ref[:,0:2]=1

    dl_old = batch_generalized_dice_loss(pred, ref)
    dl_new = batch_generalized_dice_loss_wow(pred, ref)

    print(pred)
    print(ref)
    print(dl_old==dl_new)
    print(dl_new)
    print(dl_old)

