import numpy as np
import torch

def pixel2world(x, y, z, img_width, img_height, fx, fy, cx=None, cy=None):
    """Converts image coordinates to 3D real world coordinates using depth values

    Parameters
    ----------
    x: np.array
        Array of X image coordinates.
    y: np.array
        Array of Y image coordinates.
    z: np.array
        Array of depth values for the whole image.
    img_width: int
        Width image dimension.
    img_height: int
        Height image dimension.
    fx: float
        Focal of the camera over X axis.
    fy: float
        Focal of the camera over Y axis.
    cx: float
        X coordinate of principal point of the camera.
    cy: float
        Y coordinate of principal point of the camera.

    Returns
    -------
    w_x: np.array
        Array of X world coordinates.
    w_y: np.array
        Array of Y world coordinates.
    w_z: np.array
        Array of Z world coordinates.

    """
    if cx is None:
        cx = img_width / 2
    if cy is None:
        cy = img_height / 2
    w_x = (x - cx) * z / fx
    w_y = (cy - y) * z / fy
    w_z = z
    return w_x, w_y, w_z
