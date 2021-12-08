from model.shape_spec import ShapeSpec
from model.fcoshead import FCOSHead

def build_fcoshead(cfg, input_shape=None):
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
    return FCOSHead(cfg,input_shape)