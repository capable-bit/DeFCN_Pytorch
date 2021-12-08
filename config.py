import argparse
from easydict import EasyDict

parser = argparse.ArgumentParser()
cfg = parser.parse_args()
#------------------------------------------------------------------#
cfg.NUM_DEVICES = 1
cfg.RANK = 0
cfg.NUM_WORKERS = 2
cfg.SEED = 49645663

#------------------Train-------------------------------------------------#
cfg.TRAIN = EasyDict()
cfg.TRAIN.START_EPOCH = 1
cfg.TRAIN.MAX_EPOCH = 36
cfg.TRAIN.SAVE_DIR = r"./output"
# cfg.TRAIN.RESUME = r"/home/dcxh1819/cnt/lhl/weight/model_61.pth"
cfg.TRAIN.RESUME = r"./weight/R-50.pkl"
cfg.VAL_TRAIN = 'val'
#------------------Train-------------------------------------------------#


#------------------DATA-------------------------------------------------#
cfg.DATA = EasyDict()
cfg.DATA.COCO_PATH = "/home/dcxh1819/cnt/lhl/datasets/coco/"
cfg.DATA.BATCH_SIZE = 2
#------------------DATA-------------------------------------------------#


cfg.SOLVER = EasyDict()
#------------------optimizer_scheduler-------------------------------------------------#
cfg.SOLVER.OPTIMIZER = EasyDict()
cfg.SOLVER.OPTIMIZER.BASE_LR = 0.00125
cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY = 0.0001
cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY_NORM = 0.0
cfg.SOLVER.OPTIMIZER.BIAS_LR_FACTOR = 1.0
cfg.SOLVER.OPTIMIZER.MOMENTUM = 0.9
#------------------optimizer_scheduler-------------------------------------------------#

#------------------scheduler-------------------------------------------------#
cfg.SOLVER.LR_SCHEDULER = EasyDict()
cfg.SOLVER.LR_SCHEDULER.STEPS = [1680000, 2000000]
cfg.SOLVER.LR_SCHEDULER.GAMMA = 0.1
cfg.SOLVER.LR_SCHEDULER.WARMUP_FACTOR = 0.001
cfg.SOLVER.LR_SCHEDULER.WARMUP_ITERS = 1000
cfg.SOLVER.LR_SCHEDULER.WARMUP_METHOD = 'linear'
#------------------scheduler-------------------------------------------------#


#------------------MODEL-------------------------------------------------#
cfg.MODEL = EasyDict()
cfg.MODEL.DEVICE = "cuda"
cfg.MODEL.PIXEL_MEAN = [103.53, 116.28, 123.675]
cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]

#------------------backbone----------------------------------------------#
cfg.MODEL.BACKBONE = EasyDict()
cfg.MODEL.BACKBONE.FREEZE_AT = 2

#-------------------------backbone:resnet--------------------------#
cfg.MODEL.RESNETS = EasyDict()

cfg.MODEL.RESNETS.DEPTH = 50

cfg.MODEL.RESNETS.DEEP_STEM = False

cfg.MODEL.RESNETS.NORM = 'FrozenBN'
cfg.MODEL.RESNETS.ACTIVATION = EasyDict({'NAME': 'ReLU', 'INPLACE': True})

cfg.MODEL.RESNETS.STEM_OUT_CHANNELS = 64

cfg.MODEL.RESNETS.OUT_FEATURES = ['res3', 'res4', 'res5']
cfg.MODEL.RESNETS.NUM_GROUPS = 1
cfg.MODEL.RESNETS.WIDTH_PER_GROUP = 64

cfg.MODEL.RESNETS.STEM_OUT_CHANNELS = 64
cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
cfg.MODEL.RESNETS.STRIDE_IN_1X1 = True
cfg.MODEL.RESNETS.RES5_DILATION = 1
cfg.MODEL.RESNETS.NUM_CLASSES = None
cfg.MODEL.RESNETS.ZERO_INIT_RESIDUAL = False
#-------------------------backbone:resnet--------------------------#

#-------------------------backbone:fpn-----------------------------#
cfg.MODEL.FPN = EasyDict()

cfg.MODEL.FPN.IN_FEATURES = ['res3', 'res4', 'res5']
cfg.MODEL.FPN.OUT_CHANNELS = 256
cfg.MODEL.FPN.BLOCK_IN_FEATURES = 'p5'

cfg.MODEL.FPN.FUSE_TYPE = 'sum'
#-------------------------backbone:fpn-----------------------------#

#------------------backbone---------------------------------------------#

#------------------FCOS----------------------------------------------#
cfg.MODEL.FCOS = EasyDict()
cfg.MODEL.FCOS.IN_FEATURES = ['p3', 'p4', 'p5', 'p6', 'p7']

cfg.MODEL.FCOS.NUM_CLASSES = 80
cfg.MODEL.FCOS.NUM_CONVS = 4
cfg.MODEL.FCOS.PRIOR_PROB = 0.01
cfg.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
cfg.MODEL.FCOS.NORM_REG_TARGETS = True

cfg.MODEL.FCOS.BBOX_REG_WEIGHTS = [1.0, 1.0, 1.0, 1.0]

cfg.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
cfg.MODEL.FCOS.FOCAL_LOSS_ALPHA = 0.25
cfg.MODEL.FCOS.FOCAL_LOSS_GAMMA = 2.0
cfg.MODEL.FCOS.IOU_LOSS_TYPE = 'giou'
cfg.MODEL.FCOS.REG_WEIGHT = 2.0
#--------------------FCOS----------------------------------------------#

#--------------------POTO----------------------------------------------#
cfg.MODEL.POTO = EasyDict()
cfg.MODEL.POTO.FILTER_KERNEL_SIZE = 3
cfg.MODEL.POTO.FILTER_TAU = 2
cfg.MODEL.POTO.ALPHA = 0.8
cfg.MODEL.POTO.CENTER_SAMPLING_RADIUS = 1.5
cfg.MODEL.POTO.AUX_TOPK = 9
#--------------------POTO----------------------------------------------#


#--------------------SHIFT_GENERATOR----------------------------------#
cfg.MODEL.SHIFT_GENERATOR = EasyDict()

cfg.MODEL.SHIFT_GENERATOR.NUM_SHIFTS = 1
cfg.MODEL.SHIFT_GENERATOR.OFFSET = 0.5
#--------------------SHIFT_GENERATOR----------------------------------#

#------------------MODEL-------------------------------------------------#

