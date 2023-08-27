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
from torchsummary import summary

from pyrr import Quaternion

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
    
    #print("pixels.shape", pixels.shape)
    
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
    w_y = (y - cy) * z / fy # right-handed axis !!!
    w_z = z
    return w_x, w_y, w_z


def world2pixel(x, y, z, img_width, img_height, fx, fy, cx=None, cy=None):
    if cx is None:
        cx = img_width / 2
    if cy is None:
        cy = img_height / 2
    p_x = x * fx / z + cx
    p_y = cy + y * fy / z # right-handed axis !!!
    return p_x, p_y


def points2pixels(points, img_width, img_height, fx, fy, cx=None, cy=None):
    pixels = np.zeros((points.shape[0], 2))
    pixels[:, 0], pixels[:, 1] = \
        world2pixel(points[:,0], points[:, 1], points[:, 2], img_width, img_height, fx, fy, cx, cy)
    return pixels

def nearest_hole_filling(depthmap, kernel_size=5, nearest_num=5):
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
    # default
    mask_filled = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size)))
    
    points_to_fill = (mask_filled * (depthmap == 0)).astype(np.uint8)
    points_ori = np.ones_like(points_to_fill) - points_to_fill
    
    index_list = [[i, j, abs(i) + abs(j)] 
                  for i in range(-nearest_num//2, nearest_num//2+1)
                  for j in range(-nearest_num//2, nearest_num//2+1)
                  ]
    index_list = sorted(index_list, key=lambda x : x[2]) 
    del index_list[0]
    
    depthmap_res = depthmap * points_ori
    
    for index in index_list:
        M = np.float32([[1, 0, index[0]], [0, 1, index[1]]]) # 向右平移index[0]，向下平移index[1]
        depthmap_dst = cv2.warpAffine(depthmap, M, (depthmap.shape[1], depthmap.shape[0]))
        this_fill = ((points_to_fill) * (depthmap_dst > 0)).astype(np.uint8)
        depthmap_res += depthmap_dst * this_fill
        points_to_fill -= this_fill
    
    return depthmap_res

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
    # default
    mask_filled = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size)))
    
    #mask_filled = cv2.morphologyEx(mask, cv2.MORPH_BLACKHAT, np.ones((kernel_size, kernel_size)))
    
    points_to_fill = (mask_filled * (depthmap == 0)).astype(np.uint8)
    #print("points_fo_fill", points_to_fill.shape)
    
    
    if depthmap.dtype == np.float32:
        depthmap = cv2.inpaint(depthmap, points_to_fill, 2, cv2.INPAINT_NS)
        #depthmap = cv2.inpaint(depthmap, points_to_fill, 2, cv2.INPAINT_TELEA)
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
def augment_3d(depth_intrinsic, points, depth16_img, joints_3D_Z, R2C_Mat, R2C_Trans, intrin_aug_params=dict()):
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

        fx=depth_intrinsic[0, 0]
        fy=depth_intrinsic[1, 1]
        cx=depth_intrinsic[0, 2]
        cy=depth_intrinsic[1, 2] 
        
        if intrin_aug_params:
            noise_f = 2 * np.random.random() - 1
            noise_cx = 2 * np.random.random() - 1
            noise_cy = 2 * np.random.random() - 1

            aug_degree = noise_f * (intrin_aug_params["f_up"] - intrin_aug_params["f_low"]) + intrin_aug_params["f_low"]
            fx = depth16_img.shape[1] / 2 / np.tan(aug_degree)
            fy = fx
            
            cx += noise_cx * intrin_aug_params["cx"]
            cy += noise_cy * intrin_aug_params["cy"]
        
        depth16_img = pointcloud2depthmap(points, depth16_img.shape[1], depth16_img.shape[0],
                                          fx=fx,
                                          fy=fy,
                                          cx=cx,
                                          cy=cy).astype(depth16_img.dtype)
                                          
        # depth16_img = hole_filling(depth16_img, kernel_size=2)
#        for i in range(3):                      
#            depth16_img = nearest_hole_filling(depth16_img, kernel_size=2, nearest_num=4)
        
        # R2C_Mat, R2C_Trans
        R2C_Mat_after_aug = rot @ R2C_Mat
        R2C_Trans_after_aug = rot @ R2C_Trans + tr + (np.eye(3) - rot) @ points_mean
        
        depth_intrinsic[0, 0] = fx
        depth_intrinsic[1, 1] = fy
        depth_intrinsic[0, 2] = cx
        depth_intrinsic[1, 2] = cy
        

    return depth_intrinsic, depth16_img, joints_3D_Z, R2C_Mat_after_aug, R2C_Trans_after_aug

def augment_3d_diff(joints_3D_Z, R2C_Mat, R2C_Trans):
    if len(joints_3D_Z) > 0:
        points_mean = joints_3D_Z.mean(axis=0)
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
        joints_3D_Z = joints_3D_Z + tr

        # from pointcloud to depthmap
        joints_3D_Z += points_mean


        R2C_Mat_after_aug = rot @ R2C_Mat
        R2C_Trans_after_aug = rot @ R2C_Trans + tr + (np.eye(3) - rot) @ points_mean

    return joints_3D_Z, R2C_Mat_after_aug, R2C_Trans_after_aug


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
        max_value = 8000
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


def init_spdh_model(model, cfg, device):
    #print(model.state_dict().keys())
    state_dict = {}
    spdh_checkpoint = torch.load(cfg["TRAINED_SPDH_NET_PATH"], map_location=device)["model"]
    simplenet_checkpoint = torch.load(cfg["TRAINED_SIMPLE_NET_PATH"], map_location=device)["model"]
    
    for key, value in spdh_checkpoint.items():
        if key[:6] == "module":
            new_key = "backbone" + key[6:]
            state_dict[new_key] = value
        else:
            state_dict[key] = value
    
    for key, value in simplenet_checkpoint.items():
        if key[:6] == "module":
            new_key = "simplenet" + key[6:]
        elif "simplenet" not in key:
            new_key = "simplenet." + key
        else:
            new_key = key
        state_dict[new_key] = value
    
#    for name, param in model.named_parameters():
#        print(name)
    #summary(model, (3, 128, 128))
    for k in model.state_dict():
        if "soft" in k:
            print(k)
        if k not in state_dict:
            print(f"Unloading {k}")
    model.load_state_dict(state_dict, strict=False)
    print("Loading Successfully")
    #print("spdh", spdh_checkpoint["model"].keys())
    #print("simplenet", simplenet_checkpoint["model"].keys())
    

def load_spdh_model(model, optimizer, scheduler, weights_dir,device):
    # weights_dir : xxxx/model.pth

    print(f'restoring checkpoint {weights_dir}')

    #checkpoint = torch.load(weights_dir, map_location=device)
    checkpoint = torch.load(weights_dir)
    print(checkpoint.keys())

    start_epoch = checkpoint["epoch"]
    if "global_iter" in checkpoint:
        global_iter = checkpoint["global_iter"]
    
    state_dict = {}
    for k in checkpoint["model"]:
        if k in model.state_dict():
            state_dict[k] = checkpoint["model"][k]
    #ret = model.load_state_dict(checkpoint["model"], strict=False)
    ret = model.load_state_dict(state_dict, strict=True)
    print(f'restored "{weights_dir}" model. Key errors:')
    print(ret)
    
    try:
        optimizer.load_state_dict(checkpoint["optimizer"])
    except:
        pass
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

def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])

def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))

def compute_3n_loss_42_rob(Angles, device=torch.device("cpu")):
    # R : B x 3 x 3
    # T : B x 3 x 1
    # Angles : B x N
    B, M, _ = Angles.shape
    ori_trans_list = torch.from_numpy(np.array([[0.0,0.0,0.0], [0.0,0.0,0.333], [0.0,0.0,0.0],\
                                        [0.0,-0.316,0.0], [0.0825,0.0,0.0],[-0.0825,0.384,0.0],\
                                        [0.0,0.0,0.0], [0.088,0.0,0.0],[0.0,0.0,0.107],\
                                        [0.0,0.0,0.0584], [0.0,0.0,0.0]
                                        ])).float().to(device)
    ori_angles_list = torch.from_numpy(np.array([[0.0,0.0,0.0], [0.0,0.0,0.0], [-1.5707963267948966,0.0,0.0],\
              [1.5707963267948966,0.0,0.0], [1.5707963267948966,0.0,0.0], [-1.5707963267948966,0.0,0.0],\
               [1.5707963267948966,0.0,0.0],  [1.5707963267948966,0.0,0.0], [0.0,0.0,-1.5707963267948966/2],\
               [0.0,0.0,0.0], [0.0,0.0,0.0]
                                        ])).float().to(device)    
    
    joints_info = [
              # base and plus 3 pts
              {"base" : 0, "offset" : [0.0, 0.0, 0.0]},
              {"base" : 0, "offset" : [0.0, 0.0, 0.14]}, # panda_joint1
              {"base" : 0, "offset" : [-0.11, 0.0, 0.0]},
              
              
              {"base" : 1, "offset" : [0.0, 0.0, 0.0]}, # panda_joint2
              {"base" : 1, "offset" : [0.0, -0.1294, -0.0]}, 
              
              
              {"base" : 2, "offset" : [0.0, -0.1940, 0.0]}, # panda_joint3 
              
              {"base" : 4, "offset" : [0.0, 0.0, 0.0]},
              {"base" : 4, "offset" : [0.0, 0.0, 0.1111]},
              {"base" : 4, "offset" : [0.0, 0.1240, 0.0]},
              
              
              {"base" : 5, "offset" : [0.0, 0.1299, 0.0]},
              
              
              {"base" :6, "offset" : [0.0, 0.0, 0.0583]},
              {"base" :6, "offset" : [0.088, 0.0, 0.0]}, 
              
              {"base" : 7, "offset" : [0.0, 0.0, 0.1520]},
              {"base" : 7, "offset" : [0.06, 0.06, 0.1520]}
              ]
    ori_mat = torch.eye(3).repeat(B,1).reshape(B,3,3).float().to(device)
    ori_trans = torch.zeros(B, 3, 1).float().to(device)
    ori_angles = euler_angles_to_matrix(ori_angles_list, convention="XYZ")
    
    
    kps_list, R2C_list = [], []
    for j in range(11):
        if j == 0:
            kps_list.append(ori_trans.clone())
            R2C_list.append(ori_mat.clone())
            continue
        this_mat = ori_angles[j].repeat(B, 1).reshape(B, 3, 3)
        trans = ori_trans_list[j].unsqueeze(0).repeat(B, 1)
        if 1 <= j <= 7:
            new_mat = _axis_angle_rotation("Z", Angles[:, j-1, 0])
            #print("this_mat.shape", this_mat.shape)
            #print("new_mat.shape", new_mat.shape)
            this_mat = torch.bmm(this_mat, new_mat)
#        if j == 9:
#            trans = trans + Angles[:, -1:,0] * (torch.tensor([[0.0, 1.0, 0.0]]).to(device)).repeat(B,1)
#        if j == 10:
#            trans = trans + (-2) * Angles[:, -1:,0] * (torch.tensor([[0.0, 1.0, 0.0]]).to(device)).repeat(B,1)
         
         
        ori_trans += torch.bmm(ori_mat, trans.unsqueeze(2))
        ori_mat = torch.bmm(ori_mat, this_mat)
        kps_list.append(ori_trans.float().clone())
        R2C_list.append(ori_mat.float().clone())
        
    N = len(joints_info)    
    joints_x3d_rob = torch.zeros(B, N, 3, 1).to(device)
    for idx in range(len(joints_info)):
        base_idx, offset = joints_info[idx]["base"], torch.from_numpy(np.array(joints_info[idx]["offset"]))[None, :, None].repeat(B, 1, 1).to(device)
        this_x3d = torch.bmm(R2C_list[base_idx], offset.float()) + kps_list[base_idx]
        joints_x3d_rob[:, idx, :, :] = this_x3d.clone()
    
    joints_x3d_rob = joints_x3d_rob.squeeze(3)
    return joints_x3d_rob

def compute_3n_loss_42_cam(R, T, Angles, device=torch.device("cpu")):
    # R : B x 3 x 3
    # T : B x 3 x 1
    # Angles : B x N
    B, M, _ = Angles.shape
    ori_trans_list = torch.from_numpy(np.array([[0.0,0.0,0.0], [0.0,0.0,0.333], [0.0,0.0,0.0],\
                                        [0.0,-0.316,0.0], [0.0825,0.0,0.0],[-0.0825,0.384,0.0],\
                                        [0.0,0.0,0.0], [0.088,0.0,0.0],[0.0,0.0,0.107],\
                                        [0.0,0.0,0.0584], [0.0,0.0,0.0]
                                        ])).float().to(device)
    ori_angles_list = torch.from_numpy(np.array([[0.0,0.0,0.0], [0.0,0.0,0.0], [-1.5707963267948966,0.0,0.0],\
              [1.5707963267948966,0.0,0.0], [1.5707963267948966,0.0,0.0], [-1.5707963267948966,0.0,0.0],\
               [1.5707963267948966,0.0,0.0],  [1.5707963267948966,0.0,0.0], [0.0,0.0,-1.5707963267948966/2],\
               [0.0,0.0,0.0], [0.0,0.0,0.0]
                                        ])).float().to(device)
#    joints_info = [
#            # base and plus 3 pts
#            {"base" : 0, "offset" : [0.0, 0.0, 0.0]},
#            {"base" : 0, "offset" : [0.0, 0.0, 0.1]},
#            {"base" : 0, "offset" : [0.1, 0.0, 0.0]},
#            
#            # joint1 and plus 3 pts
#            {"base" : 1, "offset" : [0.0, 0.0, -0.1920]},
#            {"base" : 1, "offset" : [0.0, 0.0, -0.0920]},
#            {"base" : 1, "offset" : [0.1, 0.0, -0.1920]},
#            
#            # joint2 and plus 3 pt
#            {"base" : 2, "offset" : [0.0, 0.0, 0.0]},
#            {"base" : 2, "offset" : [0.0, 0.0, 0.1]},
#            {"base" : 2, "offset" : [0.1, 0.0, 0.0]},
#            
#            # joint3 and plus 3 pt
#            {"base" : 3, "offset" : [0.0, 0.0, -0.1210]},
#            {"base" : 3, "offset" : [0.0, 0.0, -0.0210]},
#            {"base" : 3, "offset" : [0.1, 0.0, -0.1210]},
#            
#            # joint4 and plus 3 pt
#            {"base" : 4, "offset" : [0.0, 0.0, 0.0]},
#            {"base" : 4, "offset" : [0.0, 0.0, 0.1]},
#            {"base" : 4, "offset" : [0.1, 0.0, 0.0]},
#            
#            
#            # joint5 and plus 3 pt
#            {"base" : 5, "offset" : [0.0, 0.0, -0.2590]},
#            {"base" : 5, "offset" : [0.0, 0.0, -0.1590]},
#            {"base" : 5, "offset" : [0.1, 0.0, -0.2590]},
#            
#            
#            # joint6 and plus 3 pt
#            {"base" :6, "offset" : [0.0, 0.0, -0.0148]},
#            {"base" :6, "offset" : [0.0, 0.0, 0.0852]},
#            {"base" :6, "offset" : [0.1, 0.0, -0.0148]},
#            
#            
#            # joint7 and plus 3 pt
#            {"base" : 7, "offset" : [0.0, 0.0, 0.0520]},
#            {"base" : 8, "offset" : [0.0, 0.0, 0.0584]},
#            {"base" : 10, "offset" : [0.0, 0.0, 0.0]},
#            ]
    
    
    joints_info = [
              # base and plus 3 pts
              {"base" : 0, "offset" : [0.0, 0.0, 0.0]},
              {"base" : 0, "offset" : [0.0, 0.0, 0.14]}, # panda_joint1
              {"base" : 0, "offset" : [-0.11, 0.0, 0.0]},
              
              
              {"base" : 1, "offset" : [0.0, 0.0, 0.0]}, # panda_joint2
              {"base" : 1, "offset" : [0.0, -0.1294, -0.0]}, 
              
              
              {"base" : 2, "offset" : [0.0, -0.1940, 0.0]}, # panda_joint3 
              
              {"base" : 4, "offset" : [0.0, 0.0, 0.0]},
              {"base" : 4, "offset" : [0.0, 0.0, 0.1111]},
              {"base" : 4, "offset" : [0.0, 0.1240, 0.0]},
              
              
              {"base" : 5, "offset" : [0.0, 0.1299, 0.0]},
              
              
              {"base" :6, "offset" : [0.0, 0.0, 0.0583]},
              {"base" :6, "offset" : [0.088, 0.0, 0.0]}, 
              
              {"base" : 7, "offset" : [0.0, 0.0, 0.1520]},
              {"base" : 7, "offset" : [0.06, 0.06, 0.1520]}
              ]
    ori_mat = torch.eye(3).repeat(B,1).reshape(B,3,3).float().to(device)
    ori_trans = torch.zeros(B, 3, 1).float().to(device)
    ori_angles = euler_angles_to_matrix(ori_angles_list, convention="XYZ")
    
    
    kps_list, R2C_list = [], []
    for j in range(11):
        if j == 0:
            kps_list.append(ori_trans.clone())
            R2C_list.append(ori_mat.clone())
            continue
        this_mat = ori_angles[j].repeat(B, 1).reshape(B, 3, 3)
        trans = ori_trans_list[j].unsqueeze(0).repeat(B, 1)
        if 1 <= j <= 7:
            new_mat = _axis_angle_rotation("Z", Angles[:, j-1, 0])
            #print("this_mat.shape", this_mat.shape)
            #print("new_mat.shape", new_mat.shape)
            this_mat = torch.bmm(this_mat, new_mat)
#        if j == 9:
#            trans = trans + Angles[:, -1:,0] * (torch.tensor([[0.0, 1.0, 0.0]]).to(device)).repeat(B,1)
#        if j == 10:
#            trans = trans + (-2) * Angles[:, -1:,0] * (torch.tensor([[0.0, 1.0, 0.0]]).to(device)).repeat(B,1)
         
         
        ori_trans += torch.bmm(ori_mat, trans.unsqueeze(2))
        ori_mat = torch.bmm(ori_mat, this_mat)
        kps_list.append(ori_trans.float().clone())
        R2C_list.append(ori_mat.float().clone())


        
    N = len(joints_info)    
    joints_x3d_rob = torch.zeros(B, N, 3, 1).to(device)
    for idx in range(len(joints_info)):
        base_idx, offset = joints_info[idx]["base"], torch.from_numpy(np.array(joints_info[idx]["offset"]))[None, :, None].repeat(B, 1, 1).to(device)
        this_x3d = torch.bmm(R2C_list[base_idx], offset.float()) + kps_list[base_idx]
        joints_x3d_rob[:, idx, :, :] = this_x3d.clone()
    
    joints_x3d_rob = joints_x3d_rob.squeeze(3)
    joints_x3d_cam= torch.bmm(R, joints_x3d_rob.permute(0, 2, 1).contiguous()) + T
    return joints_x3d_cam.permute(0,2,1).contiguous()  

    
def compute_3n_loss_39(Angles, device=torch.device("cpu")): 
    # R : B x 3 x 3
    # T : B x 3 x 1
    # Angles : B x N
    B, M, _ = Angles.shape
    ori_trans_list = torch.from_numpy(np.array([[0.0,0.0,0.0], [0.0,0.0,0.333], [0.0,0.0,0.0],\
                                        [0.0,-0.316,0.0], [0.0825,0.0,0.0],[-0.0825,0.384,0.0],\
                                        [0.0,0.0,0.0], [0.088,0.0,0.0],[0.0,0.0,0.107],\
                                        [0.0,0.0,0.0584], [0.0,0.0,0.0]
                                        ])).float().to(device)
    ori_angles_list = torch.from_numpy(np.array([[0.0,0.0,0.0], [0.0,0.0,0.0], [-1.5707963267948966,0.0,0.0],\
              [1.5707963267948966,0.0,0.0], [1.5707963267948966,0.0,0.0], [-1.5707963267948966,0.0,0.0],\
               [1.5707963267948966,0.0,0.0],  [1.5707963267948966,0.0,0.0], [0.0,0.0,-1.5707963267948966/2],\
               [0.0,0.0,0.0], [0.0,0.0,0.0]
                                        ])).float().to(device)
    joints_info = [
            # base and plus 3 pts
            {"base" : 0, "offset" : [0.0, 0.0, 0.0]},
            {"base" : 0, "offset" : [0.0, 0.0, 0.1]},
            {"base" : 0, "offset" : [0.1, 0.0, 0.0]},
            
            # joint1 and plus 3 pts
            {"base" : 1, "offset" : [0.0, 0.0, -0.1920]},
            {"base" : 1, "offset" : [0.0, 0.0, -0.0920]},
            {"base" : 1, "offset" : [0.1, 0.0, -0.1920]},
            
            # joint2 and plus 3 pt
            {"base" : 2, "offset" : [0.0, 0.0, 0.0]},
            {"base" : 2, "offset" : [0.0, 0.0, 0.1]},
            {"base" : 2, "offset" : [0.1, 0.0, 0.0]},
            
            # joint3 and plus 3 pt
            {"base" : 3, "offset" : [0.0, 0.0, -0.1210]},
            {"base" : 3, "offset" : [0.0, 0.0, -0.0210]},
            {"base" : 3, "offset" : [0.1, 0.0, -0.1210]},
            
            # joint4 and plus 3 pt
            {"base" : 4, "offset" : [0.0, 0.0, 0.0]},
            {"base" : 4, "offset" : [0.0, 0.0, 0.1]},
            {"base" : 4, "offset" : [0.1, 0.0, 0.0]},
            
            
            # joint5 and plus 3 pt
            {"base" : 5, "offset" : [0.0, 0.0, -0.2590]},
            {"base" : 5, "offset" : [0.0, 0.0, -0.1590]},
            {"base" : 5, "offset" : [0.1, 0.0, -0.2590]},
            
            
            # joint6 and plus 3 pt
            {"base" :6, "offset" : [0.0, 0.0, -0.0148]},
            {"base" :6, "offset" : [0.0, 0.0, 0.0852]},
            {"base" :6, "offset" : [0.1, 0.0, -0.0148]},
            
            
            # joint7 and plus 3 pt
            {"base" : 7, "offset" : [0.0, 0.0, 0.0520]},
            {"base" : 8, "offset" : [0.0, 0.0, 0.0584]},
            {"base" : 10, "offset" : [0.0, 0.0, 0.0]},
            ]
    
    
    
    ori_mat = torch.eye(3).repeat(B,1).reshape(B,3,3).float().to(device)
    ori_trans = torch.zeros(B, 3, 1).float().to(device)
    ori_angles = euler_angles_to_matrix(ori_angles_list, convention="XYZ")
    
    
    kps_list, R2C_list = [], []
    for j in range(11):
        if j == 0:
            kps_list.append(ori_trans.clone())
            R2C_list.append(ori_mat.clone())
            continue
        this_mat = ori_angles[j].repeat(B, 1).reshape(B, 3, 3)
        trans = ori_trans_list[j].unsqueeze(0).repeat(B, 1)
        if 1 <= j <= 7:
            new_mat = _axis_angle_rotation("Z", Angles[:, j-1, 0])
            #print("this_mat.shape", this_mat.shape)
            #print("new_mat.shape", new_mat.shape)
            this_mat = torch.bmm(this_mat, new_mat)
#        if j == 9:
#            trans = trans + Angles[:, -1:,0] * (torch.tensor([[0.0, 1.0, 0.0]]).to(device)).repeat(B,1)
#        if j == 10:
#            trans = trans + (-2) * Angles[:, -1:,0] * (torch.tensor([[0.0, 1.0, 0.0]]).to(device)).repeat(B,1)
         
         
        ori_trans += torch.bmm(ori_mat, trans.unsqueeze(2))
        ori_mat = torch.bmm(ori_mat, this_mat)
        kps_list.append(ori_trans.float().clone())
        R2C_list.append(ori_mat.float().clone())
        
    N = len(joints_info)    
    joints_x3d_rob = torch.zeros(B, N, 3, 1).to(device)
    for idx in range(len(joints_info)):
        base_idx, offset = joints_info[idx]["base"], torch.from_numpy(np.array(joints_info[idx]["offset"]))[None, :, None].repeat(B, 1, 1).to(device)
        this_x3d = torch.bmm(R2C_list[base_idx], offset.float()) + kps_list[base_idx]
        joints_x3d_rob[:, idx, :, :] = this_x3d.clone()
    
    joints_x3d_rob = joints_x3d_rob.squeeze(3)
    
    return joints_x3d_rob
    
def compute_3n_loss_40(Angles, device=torch.device("cpu")):
    # R : B x 3 x 3
    # T : B x 3 x 1
    # Angles : B x N
    B, M, _ = Angles.shape
    ori_trans_list = torch.from_numpy(np.array([[0.0,0.0,0.0], [0.0,0.0,0.333], [0.0,0.0,0.0],\
                                        [0.0,-0.316,0.0], [0.0825,0.0,0.0],[-0.0825,0.384,0.0],\
                                        [0.0,0.0,0.0], [0.088,0.0,0.0],[0.0,0.0,0.107],\
                                        [0.0,0.0,0.0584], [0.0,0.0,0.0]
                                        ])).float().to(device)
    ori_angles_list = torch.from_numpy(np.array([[0.0,0.0,0.0], [0.0,0.0,0.0], [-1.5707963267948966,0.0,0.0],\
              [1.5707963267948966,0.0,0.0], [1.5707963267948966,0.0,0.0], [-1.5707963267948966,0.0,0.0],\
               [1.5707963267948966,0.0,0.0],  [1.5707963267948966,0.0,0.0], [0.0,0.0,-1.5707963267948966/2],\
               [0.0,0.0,0.0], [0.0,0.0,0.0]
                                        ])).float().to(device)
#    joints_info = [
#            # base and plus 3 pts
#            {"base" : 0, "offset" : [0.0, 0.0, 0.0]},
#            {"base" : 0, "offset" : [0.0, 0.0, 0.1]},
#            {"base" : 0, "offset" : [0.1, 0.0, 0.0]},
#            
#            # joint1 and plus 3 pts
#            {"base" : 1, "offset" : [0.0, 0.0, -0.1920]},
#            {"base" : 1, "offset" : [0.0, 0.0, -0.0920]},
#            {"base" : 1, "offset" : [0.1, 0.0, -0.1920]},
#            
#            # joint2 and plus 3 pt
#            {"base" : 2, "offset" : [0.0, 0.0, 0.0]},
#            {"base" : 2, "offset" : [0.0, 0.0, 0.1]},
#            {"base" : 2, "offset" : [0.1, 0.0, 0.0]},
#            
#            # joint3 and plus 3 pt
#            {"base" : 3, "offset" : [0.0, 0.0, -0.1210]},
#            {"base" : 3, "offset" : [0.0, 0.0, -0.0210]},
#            {"base" : 3, "offset" : [0.1, 0.0, -0.1210]},
#            
#            # joint4 and plus 3 pt
#            {"base" : 4, "offset" : [0.0, 0.0, 0.0]},
#            {"base" : 4, "offset" : [0.0, 0.0, 0.1]},
#            {"base" : 4, "offset" : [0.1, 0.0, 0.0]},
#            
#            
#            # joint5 and plus 3 pt
#            {"base" : 5, "offset" : [0.0, 0.0, -0.2590]},
#            {"base" : 5, "offset" : [0.0, 0.0, -0.1590]},
#            {"base" : 5, "offset" : [0.1, 0.0, -0.2590]},
#            
#            
#            # joint6 and plus 3 pt
#            {"base" :6, "offset" : [0.0, 0.0, -0.0148]},
#            {"base" :6, "offset" : [0.0, 0.0, 0.0852]},
#            {"base" :6, "offset" : [0.1, 0.0, -0.0148]},
#            
#            
#            # joint7 and plus 3 pt
#            {"base" : 7, "offset" : [0.0, 0.0, 0.0520]},
#            {"base" : 8, "offset" : [0.0, 0.0, 0.0584]},
#            {"base" : 10, "offset" : [0.0, 0.0, 0.0]},
#            ]
    
    
    joints_info = [
              # base and plus 3 pts
              {"base" : 0, "offset" : [0.0, 0.0, 0.0]},
              {"base" : 0, "offset" : [0.0, 0.0, 0.1]},
              {"base" : 0, "offset" : [0.1, 0.0, 0.0]},
              
              # joint1 and plus 3 pts
              {"base" : 1, "offset" : [0.0, 0.0, -0.1920]},
              {"base" : 1, "offset" : [0.0, 0.0, -0.0920]},
              {"base" : 1, "offset" : [0.1, 0.0, -0.1920]},
              
              # joint2 and plus 3 pt
              {"base" : 2, "offset" : [0.0, 0.0, 0.0]},
              {"base" : 2, "offset" : [0.0, 0.0, 0.1]},
              {"base" : 2, "offset" : [0.1, 0.0, 0.0]},
              
              # joint3 and plus 3 pt
              {"base" : 3, "offset" : [0.0, 0.0, -0.1210]},
              {"base" : 3, "offset" : [0.0, 0.0, -0.0210]},
              {"base" : 3, "offset" : [0.1, 0.0, -0.1210]},
              
              # joint4 and plus 3 pt
              {"base" : 4, "offset" : [0.0, 0.0, 0.0]},
              {"base" : 4, "offset" : [0.0, 0.0, 0.1]},
              {"base" : 4, "offset" : [0.1, 0.0, 0.0]},
              
              
              # joint5 and plus 3 pt
              {"base" : 5, "offset" : [0.0, 0.0, -0.2590]},
              {"base" : 5, "offset" : [0.0, 0.0, -0.1590]},
              {"base" : 5, "offset" : [0.1, 0.0, -0.2590]},
              
              
              # joint6 and plus 3 pt
              {"base" :6, "offset" : [0.0, 0.0, -0.0148]},
              {"base" :6, "offset" : [0.0, 0.0, 0.0852]},
              {"base" :6, "offset" : [0.1, 0.0, -0.0148]},
              
              
              # joint7 and plus 3 pt
              {"base" : 7, "offset" : [0.0, 0.0, 0.0520]},
              {"base" : 7, "offset" : [0.0, 0.0, 0.1520]},
              {"base" : 7, "offset" : [0.1, 0.0, 0.0520]}
              ]
    ori_mat = torch.eye(3).repeat(B,1).reshape(B,3,3).float().to(device)
    ori_trans = torch.zeros(B, 3, 1).float().to(device)
    ori_angles = euler_angles_to_matrix(ori_angles_list, convention="XYZ")
    
    
    kps_list, R2C_list = [], []
    for j in range(11):
        if j == 0:
            kps_list.append(ori_trans.clone())
            R2C_list.append(ori_mat.clone())
            continue
        this_mat = ori_angles[j].repeat(B, 1).reshape(B, 3, 3)
        trans = ori_trans_list[j].unsqueeze(0).repeat(B, 1)
        if 1 <= j <= 7:
            new_mat = _axis_angle_rotation("Z", Angles[:, j-1, 0])
            #print("this_mat.shape", this_mat.shape)
            #print("new_mat.shape", new_mat.shape)
            this_mat = torch.bmm(this_mat, new_mat)
        if j == 9:
            trans = trans + Angles[:, -1:,0] * (torch.tensor([[0.0, 1.0, 0.0]]).to(device)).repeat(B,1)
        if j == 10:
            trans = trans + (-2) * Angles[:, -1:,0] * (torch.tensor([[0.0, 1.0, 0.0]]).to(device)).repeat(B,1)
         
         
        ori_trans += torch.bmm(ori_mat, trans.unsqueeze(2))
        ori_mat = torch.bmm(ori_mat, this_mat)
        kps_list.append(ori_trans.float().clone())
        R2C_list.append(ori_mat.float().clone())
        
    N = len(joints_info)    
    joints_x3d_rob = torch.zeros(B, N, 3, 1).to(device)
    for idx in range(len(joints_info)):
        base_idx, offset = joints_info[idx]["base"], torch.from_numpy(np.array(joints_info[idx]["offset"]))[None, :, None].repeat(B, 1, 1).to(device)
        this_x3d = torch.bmm(R2C_list[base_idx], offset.float()) + kps_list[base_idx]
        joints_x3d_rob[:, idx, :, :] = this_x3d.clone()
    
    joints_x3d_rob = joints_x3d_rob.squeeze(3)
    
    return joints_x3d_rob
    
def compute_kps_joints_loss(R, T, Angles, device):
    # R : B x 3 x 3
    # T : B x 3 x 1
    # Angles : B x 7 x 1
    B, _, _ = Angles.shape
    ori_trans_list = torch.from_numpy(np.array([[0.0,0.0,0.0], [0.0,0.0,0.333], [0.0,0.0,0.0],\
                                        [0.0,-0.316,0.0], [0.0825,0.0,0.0],[-0.0825,0.384,0.0],\
                                        [0.0,0.0,0.0], [0.088,0.0,0.0],[0.0,0.0,0.107],\
                                        [0.0,0.0,0.0584], [0.0,0.0,0.0]
                                        ])).float().to(device)
    ori_angles_list = torch.from_numpy(np.array([[0.0,0.0,0.0], [0.0,0.0,0.0], [-1.5707963267948966,0.0,0.0],\
              [1.5707963267948966,0.0,0.0], [1.5707963267948966,0.0,0.0], [-1.5707963267948966,0.0,0.0],\
               [1.5707963267948966,0.0,0.0],  [1.5707963267948966,0.0,0.0], [0.0,0.0,-1.5707963267948966/2],\
               [0.0,0.0,0.0], [0.0,0.0,0.0]
                                        ])).float().to(device)
    joints_info = [
              # joint1 and plus 3 pts
              {"base" : 0, "offset" : [0.0, 0.0, 0.14]},
              # joint2 and plus 1 pt
              {"base" : 1, "offset" : [0.0, 0.0, 0.0]},
              # joint3 and plus 1 pt
              {"base" : 3, "offset" : [0.0, 0.0, -0.1210]},
              # joint4 and plus 1 pt
              {"base" : 4, "offset" : [0.0, 0.0, 0.0]},
              # joint5 and plus 1 pt
              {"base" : 5, "offset" : [0.0, 0.0, -0.2590]},
              # joint6 and plus 1 pt
              {"base" :5, "offset" : [0.0, 0.0158, 0.0]},
              # joint7 and plus 1 pt
              {"base" : 7, "offset" : [0.0, 0.0, 0.0520]},
              ]
    
    ori_mat = torch.eye(3).repeat(B,1).reshape(B,3,3).float().to(device)
    ori_trans = torch.zeros(B, 3, 1).float().to(device)
    ori_angles = euler_angles_to_matrix(ori_angles_list, convention="XYZ")
    
    
    kps_list, R2C_list = [], []
    for j in range(ori_trans_list.shape[0]):
        if j == 0:
            kps_list.append(ori_trans.clone())
            R2C_list.append(ori_mat.clone())
            continue
        this_mat = ori_angles[j].repeat(B, 1).reshape(B, 3, 3)
        trans = ori_trans_list[j].unsqueeze(0).repeat(B, 1)
        if 1 <= j <= 7:
            new_mat = _axis_angle_rotation("Z", Angles[:, j-1, 0])
            #print("this_mat.shape", this_mat.shape)
            #print("new_mat.shape", new_mat.shape)
            this_mat = torch.bmm(this_mat, new_mat)
#        if j == 9:
#            trans = trans + Angles[:, -1:,0] * (torch.tensor([[0.0, 1.0, 0.0]]).to(device)).repeat(B,1)
#        if j == 10:
#            trans = trans + (-2) * Angles[:, -1:,0] * (torch.tensor([[0.0, 1.0, 0.0]]).to(device)).repeat(B,1)
         
        ori_trans += torch.bmm(ori_mat, trans.unsqueeze(2))
        ori_mat = torch.bmm(ori_mat, this_mat)
        kps_list.append(ori_trans.float().clone())
        R2C_list.append(ori_mat.float().clone())
        
#    N = len(joints_info)    
#    joints_x3d_rob = torch.zeros(B, N, 3, 1).to(device)
#    for idx in range(len(joints_info)):
#        base_idx, offset = joints_info[idx]["base"], torch.from_numpy(np.array(joints_info[idx]["offset"]))[None, :, None].repeat(B, 1, 1).to(device)
#        this_x3d = torch.bmm(R2C_list[base_idx], offset.float()) + kps_list[base_idx]
#        joints_x3d_rob[:, idx, :, :] = this_x3d.clone()
    
    kps_idx_lst = [0,2,3,4,6,7,8]
    N = len(kps_idx_lst)
    joints_x3d_rob = torch.zeros(B, N, 3, 1).to(device)
    for idx in range(N):
        joints_x3d_rob[:, idx, :, :] = kps_list[kps_idx_lst[idx]]
    
    
    joints_x3d_rob = joints_x3d_rob.squeeze(3)
    #print("R.shape", R.shape)
    #print("T", T.shape)
    #print("joitns_x3d_rob", joints_x3d_rob.shape)
    joints_x3d_cam= torch.bmm(R, joints_x3d_rob.permute(0, 2, 1).contiguous()) + T
    return joints_x3d_cam.permute(0,2,1).contiguous()    


def compute_DX_loss(dt_pts, gt_pts):
    B, N, _ = dt_pts.shape
    assert dt_pts.shape == gt_pts.shape
    device = dt_pts.device
    one_vec = torch.full((B, N, 1), 1).to(device).float() # B x N x 1
    X_dt = torch.bmm(dt_pts, dt_pts.permute(0, 2, 1)) # B x N x N
    X_gt = torch.bmm(gt_pts, gt_pts.permute(0, 2, 1)) # B x N x N
    diag_X_dt = torch.diagonal(X_dt, dim1=-1, dim2=-2).unsqueeze(2) # B x N x 1
    diag_X_gt = torch.diagonal(X_gt, dim1=-1, dim2=-2).unsqueeze(2) # B x N x 1

    D_dt = torch.bmm(diag_X_dt, one_vec.permute(0, 2, 1)) + \
           torch.bmm(one_vec, diag_X_dt.permute(0, 2, 1)) - \
           2 * X_dt
    D_gt = torch.bmm(diag_X_gt, one_vec.permute(0, 2, 1)) + \
           torch.bmm(one_vec, diag_X_gt.permute(0, 2, 1)) - \
           2 * X_gt
    #print(torch.tril(D_dt))
    #print(torch.sum((D_dt - D_gt) ** 2))
    return torch.norm(torch.tril(D_dt) - torch.tril(D_gt))
    
def compute_rigid_transform(a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor = None):
    """Compute rigid transforms between two point sets
    Args:
        a (torch.Tensor): ([*,] N, 3) points
        b (torch.Tensor): ([*,] N, 3) points
        weights (torch.Tensor): ([*, ] N)
    Returns:
        Transform T ([*,] 3, 4) to get from a to b, i.e. T*a = b
    """

    assert a.shape == b.shape
    assert a.shape[-1] == 3

    if weights is not None:
        assert a.shape[:-1] == weights.shape
        assert weights.min() >= 0 and weights.max() <= 1

        weights_normalized = weights[..., None] / \
                              torch.clamp_min(torch.sum(weights, dim=-1, keepdim=True)[..., None], _EPS)
        centroid_a = torch.sum(a * weights_normalized, dim=-2)
        centroid_b = torch.sum(b * weights_normalized, dim=-2)
        a_centered = a - centroid_a[..., None, :]
        b_centered = b - centroid_b[..., None, :]
        cov = a_centered.transpose(-2, -1) @ (b_centered * weights_normalized)
    else:
        centroid_a = torch.mean(a, dim=-2)
        centroid_b = torch.mean(b, dim=-2)
        a_centered = a - centroid_a[..., None, :]
        b_centered = b - centroid_b[..., None, :]
        cov = a_centered.transpose(-2, -1) @ b_centered

    # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
    # and choose based on determinant to avoid flips
    u, s, v = torch.svd(cov, some=False, compute_uv=True)
    rot_mat_pos = v @ u.transpose(-1, -2)
    v_neg = v.clone()
    v_neg[..., 2] *= -1
    rot_mat_neg = v_neg @ u.transpose(-1, -2)
    rot_mat = torch.where(torch.det(rot_mat_pos)[..., None, None] > 0, rot_mat_pos, rot_mat_neg)

    # Compute translation (uncenter centroid)
    translation = -rot_mat @ centroid_a[..., :, None] + centroid_b[..., :, None]

    transform = torch.cat((rot_mat, translation), dim=-1)
    return transform

def write_prediction_and_gt(path_meta, joints_3d_pred_gather, joints_3d_gt_gather,  
                                         pose_pred_gather, pose_gt_gather, 
                                         joints_angle_pred_gather, joints_angle_gt_gather,
                                         kps_pred_gather, kps_gt_gather
                                          ):
    # joints_3d_pred_gather : B x num_kps x 3
    # joints_angle_pred_gather : B x num_joints
    # pose_pred_gather : # B x 3 x 4
    #if not os.path.exists(path_meta):
    file_write_meta = open(path_meta, 'w')
    meta_json = dict()
    meta_json["joints_3d_pred_gather"] = joints_3d_pred_gather.flatten(1).detach().cpu().numpy().tolist()
    meta_json["joints_3d_gt_gather"] = joints_3d_gt_gather.flatten(1).detach().cpu().numpy().tolist()
    meta_json["joints_angle_pred_gather"] = joints_angle_pred_gather.detach().cpu().numpy().tolist()
    meta_json["joints_angle_gt_gather"] = joints_angle_gt_gather.detach().cpu().numpy().tolist()
    meta_json["pose_pred_gather"] = pose_pred_gather.flatten(1).detach().cpu().numpy().tolist()
    meta_json["pose_gt_gather"] = pose_gt_gather.flatten(1).detach().cpu().numpy().tolist()
    meta_json["kps_pred_gather"] = kps_pred_gather.flatten(1).detach().cpu().numpy().tolist()
    meta_json["kps_gt_gather"] = kps_gt_gather.flatten(1).detach().cpu().numpy().tolist()
    
    json_save = json.dumps(meta_json, indent=1)
    file_write_meta.write(json_save)
    file_write_meta.close()

def load_prediction_and_gt(path_meta):
    json_in = open(path_meta, 'r')
    json_data = json.load(json_in)
    
    kwargs = dict()
    B, _ = np.array(json_data["joints_3d_pred_gather"]).shape
    kwargs["joints_3d_pred_gather"] = np.array(json_data["joints_3d_pred_gather"]).reshape(B, -1, 3)
    kwargs["joints_3d_gt_gather"] = np.array(json_data["joints_3d_gt_gather"]).reshape(B, -1, 3)
    kwargs["pose_pred_gather"] = np.array(json_data["pose_pred_gather"]).reshape(B, -1, 4)
    kwargs["pose_gt_gather"] = np.array(json_data["pose_gt_gather"]).reshape(B, -1, 4)
    kwargs["joints_angle_pred_gather"] = np.array(json_data["joints_angle_pred_gather"])
    kwargs["joints_angle_gt_gather"] = np.array(json_data["joints_angle_gt_gather"])
    kwargs["kps_pred_gather"] = np.array(json_data["kps_pred_gather"]).reshape(B, -1, 3)
    kwargs["kps_gt_gather"] = np.array(json_data["kps_gt_gather"]).reshape(B, -1, 3)
    
    return kwargs

def compute_3d_error(joints_3d_gt_norm, joints_3d_pred_norm):
    # joints_3d_pred_norm : B x num_kps x 3
    
    return joints_3d_pred_norm.permute(1, 0, 2) - joints_3d_gt_norm.permute(1, 0, 2) 

def get_peak_pts(pred_uv, input_xyz, threshold=0.5):  
    # input_xyz : B x 3 x H x W
    # pred_uv : B x N x H x W (Denote the threshold)
    mask = (pred_uv > threshold)
    mask = torch.sum(mask, dim=1) # B x H x W
    mask = (mask > 0.0).unsqueeze(3) # B x H x W x 1
    return input_xyz.permute(0, 2, 3, 1) * mask # B x H x W x 3    
    
    
def augment_affine(depth_img, joints_2D, input_w, input_h, interpolation_method, aug_mode, mean=None, std=None, mask_img=None, load_bd=False):
    raw_h, raw_w = depth_img.shape[0], depth_img.shape[1]
    c = np.array([raw_w / 2., raw_h / 2.], dtype=np.float32) # (raw_w/2, raw_h/2)
    s = max(raw_h, raw_w) * 1.0
    
    aug_s, rot = 1.0, 0 # we don't do any rotation
    
    if aug_mode:
        c, aug_s = _get_aug_param(c, s, raw_w, raw_h)
        s = s * aug_s
    
    trans_input = get_affine_transform(c, s, rot, [input_w, input_h]) 
    depth_img_input = _get_input(depth_img, trans_input, input_w, input_h, mean=mean, std=std, interpolation_method=interpolation_method) # 3 x H x W
    if mask_img is not None:
        mask_img_input = _get_input(mask_img, trans_input, input_w, input_h, mean=None, std=None, interpolation_method=cv2.INTER_NEAREST)
        if load_bd:
            edge = gen_bd(mask_img,)
            edge_img_input = _get_input(edge, trans_input, input_w, input_h, mean=None, std=None, interpolation_method=cv2.INTER_NEAREST)
        else:
            edge_img_input = None
    else:
        mask_img_input = None
        edge_img_input = None
    joints_2D = affine_transform_and_clip(joints_2D, trans_input, input_w, input_h, raw_w, raw_h)
    
    return depth_img_input, joints_2D, trans_input, mask_img_input, edge_img_input

def gen_bd(label,
               edge_pad=True, 
               edge_size=4, 
               x_k_size=6,
               y_k_size=6,
               ):
        
    edge = cv2.Canny(label.astype(np.uint8), 0.1, 0.2)
    kernel = np.ones((edge_size, edge_size), np.uint8)
    if edge_pad:
        edge = edge[y_k_size:-y_k_size, x_k_size:-x_k_size]
        edge = np.pad(edge, ((y_k_size,y_k_size),(x_k_size,x_k_size)), mode='constant')
    edge = (cv2.dilate(edge, kernel, iterations=1)>50)*1.0

    return edge

def _get_input(img, trans_input, input_w, input_h, mean=None, std=None, interpolation_method=cv2.INTER_NEAREST):
    inp = cv2.warpAffine(img, trans_input, 
                        (input_w, input_h),
                        flags=interpolation_method)
    
    if mean is not None and std is not None:
        inp = (inp.astype(np.float32) / 255.)
        inp = (inp - mean) /std
    return inp

def _get_border(border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

def _get_aug_param(c, s, width, height, disturb=False):
    aug_s = np.random.choice(np.arange(0.6, 1.4, 0.1))
    w_border = _get_border(128, width)
    h_border = _get_border(128, height)
    c[0] = np.random.randint(low=w_border, high=width - w_border)
    c[1] = np.random.randint(low=h_border, high=height - h_border)
    
    return c, aug_s

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result 

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def affine_transform_and_clip(pts, t, width, height, raw_width, raw_height,mode=None):
    n_kp, _ = pts.shape
    new_ones = np.ones(n_kp).reshape(n_kp, 1)
    new_pts = np.concatenate((pts, new_ones), axis=-1)
    new_pts = np.dot(t, new_pts.T) # 得到一个2 x n_kp的矩阵
    new_pts = new_pts.T
    
    new_pts[:, 0] = np.clip(new_pts[:, 0], 0, width-1)
    new_pts[:, 1] = np.clip(new_pts[:, 1], 0, height-1)
    
    new_pts = new_pts.tolist()
    out = []
    
    # flag=  False
    for kp in range(n_kp):
        pts_x, pts_y = pts[kp][0], pts[kp][1]
        if 0.0 <= pts_x < raw_width and 0.0 <= pts_y < raw_height:
            out.append(new_pts[kp])
        else:
            out.append([0,0])
    return np.array(out)

def solve_pnp(
    canonical_points,
    projections,
    camera_K,
    method=cv2.SOLVEPNP_EPNP,
    refinement=True,
    dist_coeffs=np.array([]),
):

    n_canonial_points = len(canonical_points)
    n_projections = len(projections)
    assert (
        n_canonial_points == n_projections
    ), "Expected canonical_points and projections to have the same length, but they are length {} and {}.".format(
        n_canonial_points, n_projections
    )

    # Process points to remove any NaNs
    canonical_points_proc = []
    projections_proc = []
    for canon_pt, proj in zip(canonical_points, projections):

        if (
            canon_pt is None
            or len(canon_pt) == 0
            or canon_pt[0] is None
            or canon_pt[1] is None
            or proj is None
            or len(proj) == 0
            or proj[0] is None
            or proj[1] is None
        ):
            continue

        canonical_points_proc.append(canon_pt)
        projections_proc.append(proj)

    # Return if no valid points
    if len(canonical_points_proc) == 0:
        return False, None, None

    canonical_points_proc = np.array(canonical_points_proc)
    projections_proc = np.array(projections_proc)
    
    #print("canonical", canonical_points_proc.shape)
    #print("camera_K", camera_K.shape)
    #print("projections_proc", projections_proc.shape)

    # Use cv2's PNP solver
    try:
        pnp_retval, rvec, tvec = cv2.solvePnP(
            canonical_points_proc.reshape(canonical_points_proc.shape[0], 1, 3),
            projections_proc.reshape(projections_proc.shape[0], 1, 2),
            camera_K,
            dist_coeffs,
            flags=method,
        )

        if refinement:
            pnp_retval, rvec, tvec = cv2.solvePnP(
                canonical_points_proc.reshape(canonical_points_proc.shape[0], 1, 3),
                projections_proc.reshape(projections_proc.shape[0], 1, 2),
                camera_K,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
                useExtrinsicGuess=True,
                rvec=rvec,
                tvec=tvec,
            )
        #print("pnp_retval", pnp_retval)
        #print("rvec", rvec)
        #print("tvec", tvec)
        translation = tvec[:, 0]
        quaternion = convert_rvec_to_quaternion(rvec[:, 0])

    except:
        pnp_retval = False
        translation = None
        quaternion = None

    return pnp_retval, translation, quaternion
    
def convert_rvec_to_quaternion(rvec):
    """Convert rvec (which is log quaternion) to quaternion"""
    theta = np.sqrt(
        rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2]
    )  # in radians
    raxis = [rvec[0] / theta, rvec[1] / theta, rvec[2] / theta]

    # pyrr's Quaternion (order is XYZW), https://pyrr.readthedocs.io/en/latest/oo_api_quaternion.html
    quaternion = Quaternion.from_axis_rotation(raxis, theta)
    quaternion.normalize()
    return quaternion



def solve_pnp_ransac(
    canonical_points,
    projections,
    camera_K,
    method=cv2.SOLVEPNP_EPNP,
    inlier_thresh_px=5.0,  # this is the threshold for each point to be considered an inlier
    dist_coeffs=np.array([]),
):

    n_canonial_points = len(canonical_points)
    n_projections = len(projections)
    assert (
        n_canonial_points == n_projections
    ), "Expected canonical_points and projections to have the same length, but they are length {} and {}.".format(
        n_canonial_points, n_projections
    )

    # Process points to remove any NaNs
    canonical_points_proc = []
    projections_proc = []
    for canon_pt, proj in zip(canonical_points, projections):

        if (
            canon_pt is None
            or len(canon_pt) == 0
            or canon_pt[0] is None
            or canon_pt[1] is None
            or proj is None
            or len(proj) == 0
            or proj[0] is None
            or proj[1] is None
        ):
            continue

        canonical_points_proc.append(canon_pt)
        projections_proc.append(proj)

    # Return if no valid points
    if len(canonical_points_proc) == 0:
        print("!!!")
        return False, None, None, None

    canonical_points_proc = np.array(canonical_points_proc)
    projections_proc = np.array(projections_proc)
    #print("canonical", canonical_points_proc)

    # Use cv2's PNP solver
    try:
        pnp_retval, rvec, tvec, inliers = cv2.solvePnPRansac(
            canonical_points_proc.reshape(canonical_points_proc.shape[0], 1, 3),
            projections_proc.reshape(projections_proc.shape[0], 1, 2),
            camera_K,
            dist_coeffs,
            reprojectionError=inlier_thresh_px,
            flags=method,
        )

        translation = tvec[:, 0]
        quaternion = convert_rvec_to_quaternion(rvec[:, 0])

    except:
        pnp_retval = False
        translation = None
        quaternion = None
        inliers = None

    return pnp_retval, translation, quaternion, inliers

if __name__ == "__main__":
    device = torch.device("cpu")
    Angles = torch.tensor([
        [-0.6526082062025893,
          0.9279837857801965,
          2.551399836921973,
          -2.33985123801545,
          1.4105617980583107,
          2.125588105143108,
          1.2248684962301084,
          0.008025813124212734
         ]
        ])
    c = compute_3n_loss(Angles[:, :, None], device)
    R = torch.from_numpy(np.array([
     [
      0.9645388230978529,
      0.26324603195543156,
      -0.019139768031082402
     ],
     [
      0.03466327105203122,
      -0.19822634128022626,
      -0.9795431416481116
     ],
     [
      -0.2616548784482034,
      0.9441439887639793,
      -0.20032198538305357
     ]
    ]))[None, :, :]
    T = torch.from_numpy(np.array([
     -0.37863880349038576,
     0.2666280120354375,
     1.5995485870204855
    ]))[None, :, None]
    
    c= torch.bmm(R.float(), c.permute(0, 2, 1).contiguous().float()) + T.float()
    print(c.permute(0, 2, 1))
    print(c.permute(0, 2, 1).shape)
    
    
    

    
    
    
    
    
    
    
