import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"


import numpy as np
import torch
import sys
from PIL import Image as PILImage
import cv2
import webcolors
from numpy.core.fromnumeric import reshape
import open3d as o3d
from torch import tensor, Tensor
from tqdm import tqdm
import json
import glob

def exists_or_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    else:
        return True

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

def save_2d_pred_visual(uv_preds, uv_gts, refer_paths, save_dir,rgb_vises=None, scale=1.0, delta=0.0):
    # uv_preds : B x c x 2 
    # uv_gts : B x c x 2
    # refer_paths : list lens of B
    # all of them are np.nadarray
    uv_preds[:, :, 0] = (uv_preds[:, :, 0]) / scale
    uv_gts[:, :, 0] = (uv_gts[:, :, 0]) / scale
    uv_preds[:, :, 1] = (uv_preds[:, :, 1] - delta) / scale
    uv_gts[:, :, 1] = (uv_gts[:, :, 1] - delta) / scale
    
        
    for uv_pred, uv_gt, refer_path in tqdm(zip(uv_preds, uv_gts, refer_paths)):
        cat_id, frame_id = refer_path.split('/')[-2], refer_path.split('/')[-1].replace("json", "png")
        seg_frame_id = "seg" + frame_id
        whole_frame_id = "whole" + frame_id
        whole_mask_frame_id = "mask" + frame_id
        dir_path = os.path.join(save_dir, cat_id)
        exists_or_mkdir(dir_path)
        
        # set path
        seg_overlay_path = os.path.join(dir_path, seg_frame_id)
        whole_overlay_path = os.path.join(dir_path, whole_frame_id)
        whole_mask_overlay_path = os.path.join(dir_path, whole_mask_frame_id)
        
        # depth_vis
        #print(refer_path)
        refer_path = refer_path.replace("json", "png")
        pred_mask_path = refer_path.replace(".png", "_ours_009_mask.exr")
        
        #print(pred_mask_path)
        depth_vis = cv2.imread(refer_path)
        depth_vis = cv2.cvtColor(depth_vis,cv2.COLOR_BGR2RGB)
        
        
        mask_vis = cv2.imread(pred_mask_path, cv2.IMREAD_UNCHANGED)[:, :, 2:3]
        mask_vis = mask_vis.astype(np.uint8)
        #print("mask_)vis.shape", mask_vis.shape)
        
        
        #print("max", np.max(mask_vis))
        #print("min", np.min(mask_vis))
        #mask_vis = ((mask_vis * 255) / mask_vis.max()).astype(np.uint8)
        
        #print("max2", np.max(mask_vis))
        
        depth_mask_vis = mask_vis * depth_vis
        
        #mask_vis = np.concatenate([mask_vis,mask_vis, mask_vis], -1)
        #mask_vis_image = PILImage.fromarray(mask_vis)
        depth_vis_image = PILImage.fromarray(depth_vis)
        depth_mask_vis_image = PILImage.fromarray(depth_mask_vis)

        # produce and save seg images
#        seg_images = []
#        for n in range(len(uv_pred)):
#           image = overlay_points_on_image(depth_vis_image, [uv_pred[n], uv_gt[n]], annotation_color_dot = ["red", "green"], point_diameter=4)
#           seg_images.append(image)
#           
#        seg_img = mosaic_images(
#                   seg_images, rows=4, cols=4, inner_padding_px=10
#               )
        
        # produce and save whole images 
        image_points = uv_pred.tolist() + uv_gt.tolist()
        annotation_color_dot = ["red"] * len(uv_pred) + ["green"] * (len(uv_gt))
        whole_img = overlay_points_on_image(depth_vis_image, image_points=image_points, annotation_color_dot = annotation_color_dot, point_diameter=4)
        whole_mask_img = overlay_points_on_image(depth_mask_vis_image, image_points=image_points, annotation_color_dot = annotation_color_dot, point_diameter=4)
        # save images
        #seg_img.save(seg_overlay_path)
        whole_img.save(whole_overlay_path)
        whole_mask_img.save(whole_mask_overlay_path)


if __name__ == "__main__":
    root_path = "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/DIFF/diff_37/INFERENCE_LOGS/01420"
    save_dir = "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/depth_c2rp/visualize/visualize_imgs/"
    print("root_path", root_path)
    json_path_list = glob.glob(os.path.join(root_path, "*.json"))
    json_path_list.sort()
    print(json_path_list)
    for idx, json_path in enumerate(json_path_list[:1]):
#        if idx == 0:
#            continue
        json_in = open(json_path, 'r')
        json_data = json.load(json_in)
        print("json_path", json_path)
        print("keys", json_data.keys())
        
        uv_preds = np.array(json_data["uv_pred_list"])
        uv_gts = np.array(json_data["uv_gt_list"])
        refer_paths = json_data["depth_path_lst"]
        
        print("uv_preds.shape", uv_preds.shape)
        print("uv_gts.shape", uv_gts.shape)
        print("refer_paths", len(refer_paths))
        
        save_2d_pred_visual(uv_preds, uv_gts, refer_paths, save_dir)
        
    
    
    
    
    
    
    
    
    
    
    
    