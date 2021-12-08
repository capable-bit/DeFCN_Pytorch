import torch.nn as nn
import torch
from typing import List
from model.shape_spec import ShapeSpec
import copy

#----------shift_generator----------------#
class ShiftGenerator(nn.Module):
    # TODO: unused_arguments: cfg
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        # fmt: off
        self.num_shifts = cfg.MODEL.SHIFT_GENERATOR.NUM_SHIFTS
        self.strides    = [x.stride for x in input_shape]
        self.offset     = cfg.MODEL.SHIFT_GENERATOR.OFFSET
        # fmt: on

        self.num_features = len(self.strides)

    @property
    def num_cell_shifts(self):
        return [self.num_shifts for _ in self.strides]

    def grid_shifts(self, grid_sizes, device):
        shifts_over_all = []
        for size, stride in zip(grid_sizes, self.strides):
            shift_x, shift_y = create_grid_offsets(size, stride, self.offset, device)
            shifts = torch.stack((shift_x, shift_y), dim=1)

            shifts_over_all.append(shifts.repeat_interleave(self.num_shifts, dim=0))

        return shifts_over_all

    def forward(self, features):
        num_images = len(features[0])
        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        shifts_over_all = self.grid_shifts(grid_sizes, features[0].device)

        shifts = [copy.deepcopy(shifts_over_all) for _ in range(num_images)]
        return shifts


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