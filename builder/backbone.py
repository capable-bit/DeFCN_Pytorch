from model.resnets import BasicStem,ResNet,ResNetBlockBase,BottleneckBlock
from model.fpn import FPN,LastLevelP6P7
from utils.model_utils import make_stage

def build_resnet_backbone(cfg, input_shape):
    depth =        cfg.MODEL.RESNETS.DEPTH
    stem_width = {50: 32}[depth]

    deep_stem =    cfg.MODEL.RESNETS.DEEP_STEM
    norm =         cfg.MODEL.RESNETS.NORM
    activation =   cfg.MODEL.RESNETS.ACTIVATION
    out_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS

    stem = BasicStem(
        in_channels=input_shape,
        out_channels=out_channels,
        norm=norm,
        activation=activation,
        deep_stem=deep_stem,
        stem_width=stem_width,
    )

    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT

    if freeze_at >= 1:      # 冻结权重
        for p in stem.parameters():
            p.requires_grad = False

    # fmt: off
    out_features =       cfg.MODEL.RESNETS.OUT_FEATURES
    num_groups =         cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group =    cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    in_channels =        cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels =       cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1 =      cfg.MODEL.RESNETS.STRIDE_IN_1X1
    res5_dilation =      cfg.MODEL.RESNETS.RES5_DILATION
    num_classes =        cfg.MODEL.RESNETS.NUM_CLASSES
    zero_init_residual = cfg.MODEL.RESNETS.ZERO_INIT_RESIDUAL

    assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(res5_dilation)

    num_blocks_per_stage = {50: [3, 4, 6, 3]}[depth]
    out_stage_idx = [{"res2": 2, "res3": 3, "res4": 4, "res5": 5, "linear": 5}[f] for f in out_features]
    max_stage_idx = max(out_stage_idx)

    stages = []
    in_channels = 2 * stem_width if deep_stem else in_channels
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2

        stage_kargs = {
            "num_blocks":           num_blocks_per_stage[idx],
            "first_stride":         first_stride,
            "in_channels":          in_channels,
            "out_channels":         out_channels,
            "norm":                 norm,
            "activation":           activation,
            "bottleneck_channels":  bottleneck_channels,
            "stride_in_1x1":        stride_in_1x1,
            "dilation":             dilation,
            "num_groups":           num_groups,
            "block_class":          BottleneckBlock
        }

        blocks = make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2

        if freeze_at >= stage_idx:
            for block in blocks:
                block.freeze()

        stages.append(blocks)

    return ResNet(stem,
                  stages,
                  num_classes=num_classes,
                  out_features=out_features,
                  zero_init_residual=zero_init_residual)

def build_backbone(cfg, input_shape=None):
    bottom_up = build_resnet_backbone(cfg, input_shape)

    in_features =      cfg.MODEL.FPN.IN_FEATURES
    out_channels =     cfg.MODEL.FPN.OUT_CHANNELS
    block_in_feature = cfg.MODEL.FPN.BLOCK_IN_FEATURES

    if block_in_feature == "p5":
        in_channels_p6p7 = out_channels

    FUSE_TYPE = cfg.MODEL.FPN.FUSE_TYPE

    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm="",
        top_block=LastLevelP6P7(in_channels_p6p7, out_channels, in_feature=block_in_feature),
        fuse_type=FUSE_TYPE,
    )

    return backbone