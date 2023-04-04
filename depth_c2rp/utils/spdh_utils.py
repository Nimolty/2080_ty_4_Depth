import cv2
import numpy as np

import csv
import math
import os
from PIL import Image as PILImage

import matplotlib.pyplot as plt
from ruamel.yaml import YAML
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm import tqdm
from pyrr import Quaternion
from ruamel.yaml import YAML
import json
import random

import math
from scipy.ndimage.filters import gaussian_filter
import webcolors
from copy import deepcopy
import time
import matplotlib.pyplot as plt
import numpy as np
import torch.distributed as dist
from matplotlib import cm

def depthmap2normals(depthmap, normalize=True, keep_dims=True):
    """
    Calculate depth normals as
        normals = gF(x,y,z) = (-dF/dx, -dF/dy, 1)
    Args:
        depthmap (np.ndarray): depth map of any dtype, single channel, len(depthmap.shape) == 3
        normalize (bool): if True, normals will be normalized to have unit-magnitude
            Default: True
        keep_dims (bool):
            if True, normals shape will be equals to depthmap shape,
            if False, normals shape will be smaller than depthmap shape.
            Default: True
    Returns:
        Depth normals
    """
    depthmap = np.asarray(depthmap, np.float32)

    if keep_dims is True:
        mask = depthmap != 0
    else:
        mask = depthmap[1:-1, 1:-1] != 0

    if keep_dims is True:
        normals = np.zeros((depthmap.shape[0], depthmap.shape[1], 3), dtype=np.float32)
        normals[1:-1, 1:-1, 0] = - (depthmap[2:, 1:-1] - depthmap[:-2, 1:-1]) / 2
        normals[1:-1, 1:-1, 1] = - (depthmap[1:-1, 2:] - depthmap[1:-1, :-2]) / 2
    else:
        normals = np.zeros((depthmap.shape[0] - 2, depthmap.shape[1] - 2, 3), dtype=np.float32)
        normals[:, :, 0] = - (depthmap[2:, 1:-1] - depthmap[:-2, 1:-1]) / 2
        normals[:, :, 1] = - (depthmap[1:-1, 2:] - depthmap[1:-1, :-2]) / 2
    normals[:, :, 2] = 1

    normals[~mask] = [0, 0, 0]

    if normalize:
        div = np.linalg.norm(normals[mask], ord=2, axis=-1, keepdims=True).repeat(3, axis=-1) + 1e-12
        normals[mask] /= div

    return normals


def depthmap2points(image, fx, fy, cx=None, cy=None):
    """Converts image coordinates to 3D real world coordinates using depth values

    Parameters
    ----------
    image: np.array
        Array of depth values for the whole image.
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
    points: np.array
        Array of XYZ world coordinates.

    """
    h, w = image.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    points = np.zeros((h, w, 3), dtype=np.float32)
    points[:, :, 0], points[:, :, 1], points[:, :, 2] = pixel2world(x, y, image, w, h, fx, fy, cx, cy)
    return points


def depthmap2pointcloud(depth, fx, fy, cx=None, cy=None):
    points = depthmap2points(depth, fx, fy, cx, cy)
    points = points.reshape((-1, 3))
    return points


def pointcloud2depthmap(points, img_width, img_height, fx, fy, cx=None, cy=None, rot=None, trans=None):
    if rot is not None or trans is not None:
        raise NotImplementedError

    depthmap = np.zeros((img_height, img_width), dtype=np.float32)
    if len(points) == 0:
        return depthmap

    points = points[np.argsort(points[:, 2])]
    pixels = points2pixels(points, img_width, img_height, fx, fy, cx, cy)
    pixels = pixels.round().astype(np.int32)
    unique_pixels, indexes, counts = np.unique(pixels, return_index=True, return_counts=True, axis=0)

    mask = (unique_pixels[:, 0] >= 0) & (unique_pixels[:, 1] >= 0) & \
           (unique_pixels[:, 0] < img_width) & (unique_pixels[:, 1] < img_height)
    depth_indexes = unique_pixels[mask]
    depthmap[depth_indexes[:, 1], depth_indexes[:, 0]] = points[indexes[mask], 2]

    return depthmap


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


def world2pixel(x, y, z, img_width, img_height, fx, fy, cx=None, cy=None):
    if cx is None:
        cx = img_width / 2
    if cy is None:
        cy = img_height / 2
    p_x = x * fx / z + cx
    p_y = cy - y * fy / z
    return p_x, p_y


def points2pixels(points, img_width, img_height, fx, fy, cx=None, cy=None):
    pixels = np.zeros((points.shape[0], 2))
    pixels[:, 0], pixels[:, 1] = \
        world2pixel(points[:,0], points[:, 1], points[:, 2], img_width, img_height, fx, fy, cx, cy)
    return pixels


def hole_filling(depthmap, kernel_size=5):
    """
    Depth map (small-)hole filling
    Args:
        depthmap (np.ndarray): depth map with dtype np.uint8 (1 or 3 channels) or np.float32 (1 channel)
    Returns:
        np.ndarray: hole-filled image
    """
    orig_shape_len = len(depthmap.shape)
    assert depthmap.dtype == np.uint8 or depthmap.dtype == np.uint16 or \
           (depthmap.dtype == np.float32 and orig_shape_len == 2) or \
           (depthmap.dtype == np.float32 and orig_shape_len == 3 and depthmap.shape[2] == 1)
    assert orig_shape_len == 2 or (orig_shape_len == 3 and depthmap.shape[2] in (1, 3))

    if orig_shape_len == 3:
        depthmap = depthmap[:, :, 0]
    mask = (depthmap > 0).astype(np.uint8)
    mask_filled = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size)))
    points_to_fill = (mask_filled * (depthmap == 0)).astype(np.uint8)
    if depthmap.dtype == np.float32:
        depthmap = cv2.inpaint(depthmap, points_to_fill, 2, cv2.INPAINT_NS)
    else:
        depthmap = cv2.inpaint(depthmap, points_to_fill, 2, cv2.INPAINT_TELEA)
    if orig_shape_len == 3:
        depthmap = np.expand_dims(depthmap, -1)

    return depthmap



def overlay_points_on_image(
    image_input,
    image_points,
    image_point_names=None,
    annotation_color_dot="red",
    annotation_color_text="red",
    point_diameter=2.0,
    point_thickness=-1,
):  # any negative value means a filled point will be drawn

    # Input argument handling
    if isinstance(image_input, str):
        image = PILImage.open(image_input).convert("RGB")
    else:
        assert isinstance(
            image_input, PILImage.Image
        ), 'Expected "image_input" to be either a PIL Image or an image path, but it is "{}".'.format(
            type(image_input)
        )
        image = image_input

    if image_points is None or len(image_points) == 0:
        return image_input  # return input if no points to overlay

    n_image_points = len(image_points)
    if image_point_names:
        assert n_image_points == len(
            image_point_names
        ), "Expected the number of image point names to be the same as the number of image points."

    if isinstance(annotation_color_dot, str):
        # Replicate to all points we're annotating
        annotation_color_dot = n_image_points * [annotation_color_dot]
    else:
        # Assume we've been provided an array
        assert (
            len(annotation_color_dot) == n_image_points
        ), "Expected length of annotation_color to equal the number of image points when annotation_color is an array (dot)."

    if isinstance(annotation_color_text, str):
        # Replicate to all points we're annotating
        annotation_color_text = n_image_points * [annotation_color_text]
    else:
        # Assume we've been provided an array
        assert (
            len(annotation_color_text) == n_image_points
        ), "Expected length of annotation_color to equal the number of image points when annotation_color is an array (text)."

    if isinstance(point_diameter, float) or isinstance(point_diameter, int):
        # Replicate to all points we're annotating
        point_diameters = n_image_points * [point_diameter]
    else:
        # Assume we've been provided an array
        assert (
            len(point_diameter) == n_image_points
        ), "Expected length of point_diameter to equal the number of image points when point_diameter is an array."
        point_diameters = point_diameter

    if isinstance(point_thickness, float):
        point_thickness_to_use = int(point_thickness)
    else:
        assert isinstance(
            point_thickness, int
        ), 'Expected "point_thickness" to be either an int, but it is "{}".'.format(
            type(point_thickness)
        )
        point_thickness_to_use = point_thickness

    # Copy image
    drawn_image = np.array(image).copy() 

    # Support for sub-pixel circles using cv2!
    shift = 4
    factor = 1 << shift

    # Annotate the image
    for idx_point in range(n_image_points):
        point = image_points[idx_point]
        # Skip points that don't have values defined (e.g. passed as None or empty arrays)
        if point is None or len(point) == 0:
            continue

        point_fixedpt = (int(point[0] * factor), int(point[1] * factor))

        point_radius = point_diameters[idx_point] / 2.0
        radius_fixedpt = int(point_radius * factor)

        # Convert color to rgb tuple if it was passed as a name
        col = annotation_color_dot[idx_point]
        annot_color_dot = webcolors.name_to_rgb(col) if isinstance(col, str) else col
        col = annotation_color_text[idx_point]
        annot_color_text = webcolors.name_to_rgb(col) if isinstance(col, str) else col

        drawn_image = cv2.circle(
            drawn_image,
            point_fixedpt,
            radius_fixedpt,
            annot_color_dot,
            thickness=point_thickness_to_use,
            shift=shift,
        )

        if image_point_names:
            text_position = (int(point[0]) + 10, int(point[1]))
            # Manual adjustments for Baxter, frame 500
            # Also:  Baxter uses the following parameters:
            #        cv2.putText(drawn_image, image_point_names[idx_point], text_position, cv2.FONT_HERSHEY_SIMPLEX, 1.5, annot_color_text, 3)
            # if idx_point == 2:  # L-S1
            #     text_position = (text_position[0], text_position[1] + 20)
            # elif idx_point == 4:  # L-E1
            #     text_position = (text_position[0], text_position[1] + 25)
            # elif idx_point == 8:  # L-Hand
            #     text_position = (text_position[0], text_position[1] + 25)
            # elif idx_point == 10:  # R-S1
            #     text_position = (text_position[0], text_position[1] + 25)
            # elif idx_point == 13:  # R-W0
            #     text_position = (text_position[0], text_position[1] + 10)
            # elif idx_point == 16:  # R-Hand
            #     text_position = (text_position[0], text_position[1] + 20)
            cv2.putText(
                drawn_image,
                image_point_names[idx_point],
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                annot_color_text,
                2,
            )

    image_as_pil = PILImage.fromarray(drawn_image)

    return image_as_pil

def mosaic_images(
    image_array_input,
    rows=None,
    cols=None,
    outer_padding_px=0,
    inner_padding_px=0,
    fill_color_rgb=(255, 255, 255),
):

    # Input argument handling
    assert (
        image_array_input
        and len(image_array_input) > 0
        and not isinstance(image_array_input, str)
    ), "Expected image_array_input to be an array of image inputs, but it is {}.".format(
        type(image_array_input)
    )

    # Check whether we were provided PIL images or paths to images
    if isinstance(image_array_input[0], str):
        # Assume we're given image paths
        image_array = [
            PILImage.open(img_path).convert("RGB") for img_path in image_array_input
        ]
    else:
        # Assume we're given PIL images
        assert isinstance(
            image_array_input[0], PILImage.Image
        ), 'Expected "image_array_input" to contain either image paths or PIL Images, but it is "{}".'.format(
            type(image_array_input[0])
        )
        image_array = image_array_input

    # Verify that all images have the same resolution. Necessary for the mosaic image math to work correctly.
    n_images = len(image_array)
    image_width, image_height = image_array[0].size
    for image in image_array:
        this_image_width, this_image_height = image.size
        # print('this_image_size', image.size)
        assert (
            this_image_width == image_width and this_image_height == image_height
        ), "All images must have the same resolution."

    # Handle rows and cols inputs
    assert (
        rows or cols
    ), "Expected either rows or cols (or both) to be specified, but neither are."

    if rows:
        assert (
            isinstance(rows, int) and rows > 0
        ), "If specified, expected rows to be a positive integer, but it is {} with value {}.".format(
            type(rows), rows
        )
    else:
        # Calculate rows from cols
        rows = int(math.ceil(float(n_images) / float(cols)))

    if cols:
        assert (
            isinstance(cols, int) and cols > 0
        ), "If specified, expected cols to be a positive integer, but it is {} with value {}.".format(
            type(cols), cols
        )
    else:
        # Calculate cols from rows
        cols = int(math.ceil(float(n_images) / float(rows)))

    assert (
        rows * cols >= n_images
    ), "The number of mosaic rows and columns is too small for the number of input images."

    assert (
        isinstance(outer_padding_px, int) and outer_padding_px >= 0
    ), "Expected outer_padding_px to be an integer with value greater than or equal to zero, but it is {} with value {}".format(
        type(outer_padding_px), outer_padding_px
    )

    assert (
        isinstance(inner_padding_px, int) and inner_padding_px >= 0
    ), "Expected inner_padding_px to be an integer with value greater than or equal to zero, but it is {} with value {}".format(
        type(inner_padding_px), inner_padding_px
    )

    assert (
        len(fill_color_rgb) == 3
    ), "Expected fill_color_rgb to be a RGB array of length 3, but it has length {}.".format(
        len(fill_color_rgb)
    )

    # Construct mosaic
    mosaic = PILImage.new(
        "RGB",
        (
            cols * image_width + 2 * outer_padding_px + (cols - 1) * inner_padding_px,
            rows * image_height + 2 * outer_padding_px + (rows - 1) * inner_padding_px,
        ),
        fill_color_rgb,
    )

    # Paste images into mosaic
    img_idx = 0
    for r in range(rows):
        for c in range(cols):
            if img_idx < n_images:
                img_loc = (
                    c * image_width + outer_padding_px + c * inner_padding_px,
                    r * image_height + outer_padding_px + r * inner_padding_px,
                )
                mosaic.paste(image_array[img_idx], img_loc)
                img_idx += 1

    return mosaic

def quaternionToRotation(q):
    w, x, y, z = q
    r00 = 1 - 2 * y ** 2 - 2 * z ** 2
    r01 = 2 * x * y - 2 * w * z
    r02 = 2 * x * z + 2 * w * y

    r10 = 2 * x * y + 2 * w * z
    r11 = 1 - 2 * x ** 2 - 2 * z ** 2
    r12 = 2 * y * z - 2 * w * x

    r20 = 2 * x * z - 2 * w * y
    r21 = 2 * y * z + 2 * w * x
    r22 = 1 - 2 * x ** 2 - 2 * y ** 2
    r = [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]]
    return r


# Augmentation
def augment_3d(depth_intrinsic, points, depth16_img, joints_3D_Z):
    if len(points) > 0:
        points_mean = points.mean(axis=0)
        points -= points_mean
        joints_3D_Z -= points_mean

        p_rot = random.random()
        if p_rot < 1/3:
            # rotation around x axis
            angle = (random.random() * 2 - 1) * 5
            a = angle / 180 * np.pi
            rot = np.asarray([
                [1, 0, 0],
                [0, np.cos(a), -np.sin(a)],
                [0, np.sin(a), np.cos(a)],
            ])  # x axis
        elif p_rot >= 2/3:
            # rotation around y axis
            angle = (random.random() * 2 - 1) * 5
            a = angle / 180 * np.pi
            rot = np.asarray([
                [np.cos(a), 0, np.sin(a)],
                [0, 1, 0],
                [-np.sin(a), 0, np.cos(a)],
            ])  # y axis
        else:
            # rotation around z axis
            angle = (random.random() * 2 - 1) * 0
            a = angle / 180 * np.pi
            rot = np.asarray([
                [np.cos(a), -np.sin(a), 0],
                [np.sin(a), np.cos(a), 0],
                [0, 0, 1],
            ])  # z axis
        points = points @ rot.T
        joints_3D_Z = joints_3D_Z @ rot.T

        p_tr = random.random()
        if p_tr < 1/3:
            # translation over x axis
            tr_x = (random.random() * 2 - 1) * 0.08
            tr = np.array([tr_x, 0, 0])
        elif p_tr >= 2/3:
            # translation over y axis
            tr_y = (random.random() * 2 - 1) * 0
            tr = np.array([0, tr_y, 0])
        else:
            # translation over z axis
            tr_z = (random.random() * 2 - 1) * 0.08
            tr = np.array([0, 0, tr_z])
        points = points + tr
        joints_3D_Z = joints_3D_Z + tr

        # from pointcloud to depthmap
        points += points_mean
        joints_3D_Z += points_mean
        depth16_img = pointcloud2depthmap(points, depth16_img.shape[1], depth16_img.shape[0],
                                          fx=depth_intrinsic[0, 0],
                                          fy=depth_intrinsic[1, 1],
                                          cx=depth_intrinsic[0, 2],
                                          cy=depth_intrinsic[1, 2]).astype(depth16_img.dtype)
        depth16_img = hole_filling(depth16_img, kernel_size=2)

    return depth16_img, joints_3D_Z


# Normalization
def apply_depth_normalization_16bit_image(img, norm_type):
    """Applies normalization over 16bit depth images

    Parameters
    ----------
    img: np.array
        16bit depth image.
    norm_type: str
        Type of normalization (min_max, mean_std supported).

    Returns
    -------
    tmp: np.array
        Normalized and 16bit depth image (zero-centered in the case of mean_std normalization).

    """
    if norm_type == "min_max":
        min_value = 0
        max_value = 5000
        tmp = (img - min_value) / (max_value - min_value)
    elif norm_type == "mean_std":
        tmp = (img - img.mean()) / img.std()
    elif norm_type == "batch_mean_std":
        raise NotImplementedError
    elif norm_type == "dataset_mean_std":
        raise NotImplementedError
    else:
        raise NotImplementedError
    return tmp

# Utils
def heatmap_from_kpoints_array(kpoints_array, shape, sigma):
    """Converts N 2D keypoints to N gaussian heatmaps

    Parameters
    ----------
    kpoints_array: np.array
        Array of 2D coordinates.
    shape: tuple
        Heatmap dimension (HxW).
    sigma: float
        Variance value of the gaussian.

    Returns
    -------
    heatmaps: np.array
        Array of NxHxW gaussian heatmaps.
    """
    heatmaps = []
    for kpoint in kpoints_array:
        heatmaps.append(kpoint_to_heatmap(kpoint, shape, sigma))
    return np.stack(heatmaps, axis=-1)

def kpoint_to_heatmap(kpoint, shape, sigma):
    """Converts a 2D keypoint to a gaussian heatmap

    Parameters
    ----------
    kpoint: np.array
        2D coordinates of keypoint [x, y].
    shape: tuple
        Heatmap dimension (HxW).
    sigma: float
        Variance value of the gaussian.

    Returns
    -------
    heatmap: np.array
        A gaussian heatmap HxW.
    """
    map_h, map_w = shape
    if np.all(kpoint > 0):
        x, y = kpoint
        x *= map_w
        y *= map_h
        xy_grid = np.mgrid[:map_w, :map_h].transpose(2, 1, 0)
        heatmap = np.exp(-np.sum((xy_grid - (x, y)) ** 2, axis=-1) / sigma ** 2)
        heatmap /= (heatmap.max() + np.finfo('float32').eps)
    else:
        heatmap = np.zeros((map_h, map_w))
    return heatmap


def gkern(d, h, w, center, s=2, device='cuda'):
    x = torch.arange(0, w, 1).float().to(device)
    y = torch.arange(0, h, 1).float().to(device)
    y = y.unsqueeze(1)
    z = torch.arange(0, d, 1).float().to(device)
    z = z.unsqueeze(1).unsqueeze(1)

    x0 = center[0]
    y0 = center[1]
    z0 = center[2]

    return torch.exp(-1 * ((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) / s ** 2)


def load_spdh_model(model, optimizer, scheduler, weights_dir,device):
    # weights_dir : xxxx/model.pth

    print(f'restoring checkpoint {weights_dir}')

    checkpoint = torch.load(weights_dir, map_location=device)

    start_epoch = checkpoint["epoch"]
    if "global_iter" in checkpoint:
        global_iter = checkpoint["global_iter"]
    
    ret = model.load_state_dict(checkpoint["model"], strict=True)
    print(f'restored "{weights_dir}" model. Key errors:')
    print(ret)
    
    optimizer.load_state_dict(checkpoint["optimizer"])
    print(f'restore AdamW optimizer')
    
    scheduler.load_state_dict(checkpoint["scheduler"])
    print(f'restore AdamW scheduler')
    return model, optimizer, scheduler, start_epoch, global_iter

def reduce_mean(loss, num_gpus):
    rt = loss.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= num_gpus
    return rt

def save_weights(save_dir, epoch, global_iter, model, optimizer, scheduler, cfg):
    save_dict = {
        'epoch': epoch,
        'global_iter': global_iter,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'config': cfg
    }

    torch.save(save_dict, str(save_dir))

def visualize_inference_results(ass_add_res, ass_mAP_res, writer, epoch):
    for key, value in ass_add_res.items():
        writer.add_scalar(f"Ass_Add/{key}", value, epoch)
    for key, value in ass_mAP_res.items():
        writer.add_scalar("Ass_3D_ACC/{:.5f} m".format(float(key)), value, epoch)
