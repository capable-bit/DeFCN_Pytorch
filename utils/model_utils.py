from model.norm import FrozenBatchNorm2d
from utils.common import align_and_update_state_dicts,convert_ndarray_to_tensor
import torch.nn.functional as F
from typing import Any, List, Sequence, Tuple, Union
import os
import torch
import pickle
from torch import nn

class Shift2BoxTransform(object):
    def __init__(self, weights):
        self.weights = weights

    def get_deltas(self, shifts, boxes):
        assert isinstance(shifts, torch.Tensor), type(shifts)
        assert isinstance(boxes, torch.Tensor), type(boxes)

        deltas = torch.cat((shifts - boxes[..., :2], boxes[..., 2:] - shifts),
                           dim=-1) * shifts.new_tensor(self.weights)
        return deltas

    def apply_deltas(self, deltas, shifts):
        assert torch.isfinite(deltas).all().item()
        shifts = shifts.to(deltas.dtype)

        if deltas.numel() == 0:
            return torch.empty_like(deltas)

        deltas = deltas.view(deltas.size()[:-1] + (-1, 4)) / shifts.new_tensor(self.weights)
        boxes = torch.cat((shifts.unsqueeze(-2) - deltas[..., :2],
                           shifts.unsqueeze(-2) + deltas[..., 2:]),
                          dim=-1).view(deltas.size()[:-2] + (-1, ))
        return boxes

def get_norm(norm, out_channels):
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "FrozenBN": FrozenBatchNorm2d,
            "GN": lambda channels: nn.GroupNorm(32, channels),
            "nnSyncBN": nn.SyncBatchNorm,  # keep for debugging
        }[norm]
    return norm(out_channels)

def get_activation(activation):
    if activation is None:
        return None

    atype = activation.NAME
    inplace = activation.INPLACE
    act = {
        "ReLU": nn.ReLU,
        "ReLU6": nn.ReLU6,
    }[atype]
    return act(inplace=inplace)
def make_stage(block_class, num_blocks, first_stride, **kwargs):
    blocks = []
    for i in range(num_blocks):
        blocks.append(block_class(stride=first_stride if i == 0 else 1, **kwargs))
        kwargs["in_channels"] = kwargs["out_channels"]
    return blocks

def create_grid_offsets(size, stride, offset, device):
    grid_height, grid_width = size
    shifts_start = offset * stride
    shifts_x = torch.arange(
        shifts_start, grid_width * stride + shifts_start, step=stride,
        dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        shifts_start, grid_height * stride + shifts_start, step=stride,
        dtype=torch.float32, device=device
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    return shift_x, shift_y


# 加载权重
def resume_or_load(cfg,model):
    path = cfg.TRAIN.RESUME

    print("Loading checkpoint from {}".format(path))
    with open(path,"rb") as f:
        data = pickle.load(f, encoding="latin1")
        data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
        checkpoint = {"model": data, "__author__": "Caffe2", "matching_heuristics": True}
        convert_ndarray_to_tensor(checkpoint["model"])

    model_state_dict = model.state_dict()
    align_and_update_state_dicts(
        model_state_dict,
        checkpoint["model"],
        c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
    )

    checkpoint["model"] = model_state_dict
    checkpoint_state_dict = checkpoint.pop("model")  # super()._load_model(checkpoint)
    convert_ndarray_to_tensor(checkpoint_state_dict)
    model.load_state_dict(checkpoint_state_dict, strict=False)


def focal_loss(
    probs,
    targets,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
):
    ce_loss = F.binary_cross_entropy(
        probs, targets, reduction="none"
    )
    p_t = probs * targets + (1 - probs) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

def iou_loss(
    inputs,
    targets,
    weight=None,
    box_mode="xyxy",
    loss_type="iou",
    smooth=False,
    reduction="none"
):
    if box_mode == "ltrb":
        inputs = torch.cat((-inputs[..., :2], inputs[..., 2:]), dim=-1)
        targets = torch.cat((-targets[..., :2], targets[..., 2:]), dim=-1)
    elif box_mode != "xyxy":
        raise NotImplementedError

    eps = torch.finfo(torch.float32).eps

    inputs_area = (inputs[..., 2] - inputs[..., 0]).clamp_(min=0) \
        * (inputs[..., 3] - inputs[..., 1]).clamp_(min=0)
    targets_area = (targets[..., 2] - targets[..., 0]).clamp_(min=0) \
        * (targets[..., 3] - targets[..., 1]).clamp_(min=0)

    w_intersect = (torch.min(inputs[..., 2], targets[..., 2])
                   - torch.max(inputs[..., 0], targets[..., 0])).clamp_(min=0)
    h_intersect = (torch.min(inputs[..., 3], targets[..., 3])
                   - torch.max(inputs[..., 1], targets[..., 1])).clamp_(min=0)

    area_intersect = w_intersect * h_intersect
    area_union = targets_area + inputs_area - area_intersect
    if smooth:
        ious = (area_intersect + 1) / (area_union + 1)
    else:
        ious = area_intersect / area_union.clamp(min=eps)

    if loss_type == "iou":
        loss = -ious.clamp(min=eps).log()
    elif loss_type == "linear_iou":
        loss = 1 - ious
    elif loss_type == "giou":
        g_w_intersect = torch.max(inputs[..., 2], targets[..., 2]) \
            - torch.min(inputs[..., 0], targets[..., 0])
        g_h_intersect = torch.max(inputs[..., 3], targets[..., 3]) \
            - torch.min(inputs[..., 1], targets[..., 1])
        ac_uion = g_w_intersect * g_h_intersect
        gious = ious - (ac_uion - area_union) / ac_uion.clamp(min=eps)
        loss = 1 - gious
    else:
        raise NotImplementedError
    if weight is not None:
        loss = loss * weight.view(loss.size())
        if reduction == "mean":
            loss = loss.sum() / max(weight.sum().item(), eps)
    else:
        if reduction == "mean":
            loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()

    return loss

def sigmoid_focal_loss(
    logits,
    targets,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
):
    p = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


class ImageList(object):

    def __init__(self, tensor: torch.Tensor, image_sizes: List[Tuple[int, int]]):
        self.tensor = tensor
        self.image_sizes = image_sizes

    def __len__(self) -> int:
        return len(self.image_sizes)

    def __getitem__(self, idx: Union[int, slice]) -> torch.Tensor:
        size = self.image_sizes[idx]
        return self.tensor[idx, ..., : size[0], : size[1]]  # type: ignore

    def to(self, *args: Any, **kwargs: Any) -> "ImageList":
        cast_tensor = self.tensor.to(*args, **kwargs)
        return ImageList(cast_tensor, self.image_sizes)

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    @staticmethod
    def from_tensors(
        tensors: Sequence[torch.Tensor],
        size_divisibility: int = 0,
        pad_ref_long: bool = False,
        pad_value: float = 0.0,
    ) -> "ImageList":
        assert len(tensors) > 0
        assert isinstance(tensors, (tuple, list))
        for t in tensors:
            assert isinstance(t, torch.Tensor), type(t)
            assert t.shape[1:-2] == tensors[0].shape[1:-2], t.shape
        # per dimension maximum (H, W) or (C_1, ..., C_K, H, W) where K >= 1 among all tensors
        max_size = list(max(s) for s in zip(*[img.shape for img in tensors]))
        if pad_ref_long:
            max_size_max = max(max_size[-2:])
            max_size[-2:] = [max_size_max] * 2
        max_size = tuple(max_size)

        if size_divisibility > 0:
            import math

            stride = size_divisibility
            max_size = list(max_size)  # type: ignore
            max_size[-2] = int(math.ceil(max_size[-2] / stride) * stride)  # type: ignore
            max_size[-1] = int(math.ceil(max_size[-1] / stride) * stride)  # type: ignore
            max_size = tuple(max_size)

        image_sizes = [im.shape[-2:] for im in tensors]

        if len(tensors) == 1:
            # This seems slightly (2%) faster.
            # TODO: check whether it's faster for multiple images as well
            image_size = image_sizes[0]
            padding_size = [0, max_size[-1] - image_size[1], 0, max_size[-2] - image_size[0]]
            if all(x == 0 for x in padding_size):  # https://github.com/pytorch/pytorch/issues/31734
                batched_imgs = tensors[0].unsqueeze(0)
            else:
                padded = F.pad(tensors[0], padding_size, value=pad_value)
                batched_imgs = padded.unsqueeze_(0)
        else:
            batch_shape = (len(tensors),) + max_size
            batched_imgs = tensors[0].new_full(batch_shape, pad_value)
            for img, pad_img in zip(tensors, batched_imgs):
                pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)

        return ImageList(batched_imgs.contiguous(), image_sizes)


# 保存模型
def save_model(cfg,model,epoch):
    data = {}

    data["model"] = model.state_dict()
    basename = "model_"+str(epoch)+".pth"

    save_file = os.path.join(cfg.TRAIN.SAVE_DIR, basename)
    assert os.path.basename(save_file) == basename, basename

    print("save final model to", save_file)
    with open(save_file, "wb") as f:
        torch.save(data, f)


class weight_init:
    @staticmethod
    def c2_xavier_fill(module: nn.Module):
        nn.init.kaiming_uniform_(module.weight, a=1)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

    @staticmethod
    def c2_msra_fill(module: nn.Module):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

