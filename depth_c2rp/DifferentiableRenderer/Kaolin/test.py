import kaolin as kal
import torch
import torch.nn as nn
import math
from matplotlib import pyplot as plt
import cv2
import numpy as np
import time
import random
from PIL import Image as PILImage
#from .Render_utils import projectiveprojection_real, euler_angles_to_matrix, load_part_mesh, concat_part_mesh, exists_or_mkdir
#from .Render_utils import quaternion_to_matrix, matrix_to_quaternion, euler_angles_to_matrix, matrix_to_euler_angles, seg_and_transform, compute_rotation_matrix_from_ortho6d
from tqdm import tqdm
import argparse
import os

if __name__ == "__main__":
    path = "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/materials/franka_panda/Link6.obj"
    this_mesh = kal.io.obj.import_mesh(path, with_materials=True, with_normals=True)
    vertices = this_mesh.vertices.numpy()
    lst = []
    for vertice in vertices:
        if abs(np.sqrt((vertice[0] ** 2 + vertice[1] ** 2)) - 0.0479) < 1e-3:
            lst.append(vertice[2])
    print(np.min(lst))
