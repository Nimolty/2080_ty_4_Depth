import random
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import copy
import os
import torch
import torch.nn.functional as F
from torch.nn.parallel import DataParallel as DP
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def init_optimizer(model, cfg):
    optim_cfg = cfg["OPTIMIZER"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=optim_cfg["LR"])
    scheduler = MultiStepLR(optimizer, optim_cfg["DECAY_STEPS"], gamma=0.1)
    return optimizer, scheduler