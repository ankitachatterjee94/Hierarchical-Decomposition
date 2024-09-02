import os
from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# CUDNN
# -----------------------------------------------------------------------------
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = 'vgg16'
_C.MODEL.CKPT = '/home/ankitaC/Ankita/hsd-cnn/HSDCNN-master/hsd_semantic/checkpoints/cifar100/vgg16/checkpoint-02-03.pth.tar' 
_C.MODEL.CKPT_PATH = '/home/ankitaC/Ankita/hsd-cnn/HSDCNN-master/hsd_semantic/checkpoints' 
_C.MODEL.HSD_CKPT = ''

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.ROOT = '/home/ankitaC/Ankita/hsd-cnn/data' 
_C.DATASET.NAME = 'cifar100'
_C.DATASET.NUM_CLASSES = 100

# -----------------------------------------------------------------------------
# Clustering
# -----------------------------------------------------------------------------
_C.CLUSTER = CN()
_C.CLUSTER.DIST_METRIC = 'euclidean' 
_C.CLUSTER.LINK_METHOD = 'ward'
_C.CLUSTER.CPR = 0.5


# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.BATCH_SIZE = 256
_C.SOLVER.WORKERS = 4
_C.SOLVER.XE = False
_C.SOLVER.NUM_EPOCHS = 100
_C.SOLVER.BASE_LR = 0.1
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 5e-4
_C.SOLVER.SCHEDULER_FACTOR = 0.1
_C.SOLVER.SCHEDULER_PATIENCE = 8
_C.SOLVER.SEPARATE = False  # for hsd-cnn, consider subnet predictions separately
# -----------------------------------------------------------------------------
# Hierarchy
# -----------------------------------------------------------------------------
_C.HIERARCHY = CN()
_C.HIERARCHY.BETA = 10
_C.HIERARCHY.ROOT = '/home/ankitaC/Ankita/hsd-cnn/HSDCNN-master/hsd_semantic/hierarchy'
_C.HIERARCHY.IMPACT_PATH = '/home/ankitaC/Ankita/hsd-cnn/impact_scores'
"""
if __name__ == "__main__":
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
"""
