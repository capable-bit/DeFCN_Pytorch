from model.fcos import FCOS
from builder.backbone import build_backbone
from builder.fcoshead import build_fcoshead
from defcn_pytorch.builder.shift_generator import build_shift_generator

# ---------model------------- #
# 构建模型
def build_model(cfg):
    cfg.build_backbone = build_backbone
    cfg.build_fcoshead = build_fcoshead
    cfg.build_shift_generator = build_shift_generator
    model = FCOS(cfg)
    return model
# -----------model------------#