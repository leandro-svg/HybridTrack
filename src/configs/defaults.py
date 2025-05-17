# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .config_utils import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

########## an example
# ###### kalman filter
# _C.KALMAN = CN()
# _C.KALMAN.R = 0.6

# The version number, to upgrade from old configs to new ones if any
# changes happen. It's recommended to keep a VERSION in your config file.
_C.VERSION = 2

##### device setting
_C.DEVICE = "cuda:0"

##### model setting
_C.MODEL = CN()
_C.MODEL.TYPE = "SingleResCrossModelZ"
_C.MODEL.CLASSIFIER = False
_C.MODEL.WEIGHTS = ""
_C.MODEL.ADD_FEAT_TRANS = False
_C.MODEL.ROT_ADD_PCD = False
_C.MODEL.CENTRIC_Z_MULTIPLE_IMG_FEAT = False
_C.MODEL.CONTOUR = False
_C.MODEL.PCD_DISCRIMINATOR = True # different version of pcd discriminator
_C.MODEL.MONO_CHANNEL = False
_C.MODEL.ROTATION_MODE = 'QUATERNION'
_C.MODEL.REFINE = False
_C.MODEL.IND_FEAT = True
_C.MODEL.SAVE_PATH = ''

##### feature extractor
_C.BACKBONE = CN()
_C.BACKBONE.TYPE = "resnet50"
_C.BACKBONE.PRETRAINED = True
_C.BACKBONE.N_CLASSES = 79

##### camera
_C.CAMERA = CN()
_C.CAMERA.FX = 1331.0226
_C.CAMERA.FY = 1331.0226
_C.CAMERA.CX = 960.0
_C.CAMERA.CY = 540.0
_C.CAMERA.WIDTH = 1920
_C.CAMERA.HEIGHT = 1080

##### dataset
_C.DATASET = CN()
_C.DATASET.MODE = "train"
_C.DATASET.CLASS= "Car"
_C.DATASET.NAME = "KITTIDataset"
_C.DATASET.ROOT = "/home/leandro/Documents/HybridTrack/DeepKalTrack"
_C.DATASET.N_PCD = 1000
_C.DATASET.SET = ('city-CameraRenderSubStreet.001',)
_C.DATASET.SCENE = 'city'
_C.DATASET.AUGMENT = True
_C.DATASET.CAMERA_NAME = 'CameraRenderSubStreet.001'
_C.DATASET.RESIZE = 256
_C.DATASET.TEMPLATE_PCD = False
_C.DATASET.CROP = False
_C.DATASET.VALID_PCD = False
_C.DATASET.PADDING = False
_C.DATASET.APOLLO_LABEL = False
_C.DATASET.APOLLO_SET = ()
_C.DATASET.APOLLO_LABEL_SET = (0, )
_C.DATASET.SOURCE_STYLE_ALIGN = False
_C.DATASET.SYN_SET = ()
_C.DATASET.APOLLO_PSEUDO_SET = ()
_C.DATASET.SEQ_LEN = 20
_C.DATASET.SEQ_STRIDE = 1
_C.DATASET.RATIO_DATASET = 100
##### output
_C.OUTPUT = "result"

############ trainer
_C.TRAINER = CN()
_C.TRAINER.EPOCH = 4000
_C.TRAINER.LR = 0.001
_C.TRAINER.WD = 0.00001
_C.TRAINER.BATCH_SIZE = 128
_C.TRAINER.USE_CUDA = True
_C.TRAINER.ALPHA = 1.0
_C.TRAINER.RELATIVE_LOSS = False
_C.TRAINER.TEMPORAL_LOSS = False
_C.TRAINER.N_TEST = 100
_C.TRAINER.T_TEST = 20
_C.TRAINER.IN_MULT_LKF = 1
_C.TRAINER.OUT_MULT_LKF = 1
# Optional: add ONLINE_TRACKING if you use it
_C.TRAINER.ONLINE_TRACKING = False
_C.TRAINER.N_ITER = 5000
_C.TRAINER.LR_DECAY = 0.5
_C.TRAINER.LR_DECAY_STEP = (5000, 10000,)
_C.TRAINER.LOSS_WEIGHT_DECAY = 1000
_C.TRAINER.WARM_UP_STEP = 0
_C.TRAINER.MULTI_GPUS = False
_C.TRAINER.ETA_MIN = 1e-5
_C.TRAINER.N_E = 1000
_C.TRAINER.RANDOMINIT_TRAIN = False
_C.TRAINER.N_CV = 500
_C.TRAINER.RANDOMINIT_CV = False
_C.TRAINER.USE_FUSED_FEATURES = True

############ Loss
_C.LOSS = CN()
_C.LOSS.TYPE = ("pose", "cls", "mask", "render")
_C.LOSS.ADAPTIVE = False

########## solver
_C.SOLVER = CN()
_C.SOLVER.OPTIM = "ADAM"
_C.SOLVER.LR_SCHEDULE = "cosine"
