import numpy as np
import os
import json
import cv2

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
        inp = (inp.astype(np.float64) / 255.)
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

def affine_transform_pts(pts, t):
    n_kp, _ = pts.shape
    new_ones = np.ones(n_kp).reshape(n_kp, 1)
    new_pts = np.concatenate((pts, new_ones), axis=-1)
    new_pts = np.dot(t, new_pts.T) # 得到一个2 x n_kp的矩阵
    new_pts = new_pts.T
    
    return new_pts

def batch_affine_transform_pts(pts, t):
    # pts : B x n_kp x 2
    # t : B x 2 x 3
    b, n_kp, _ = pts.shape
    new_ones = np.ones((b, n_kp, 1))
    new_pts = np.concatenate((pts, new_ones), axis=-1)
    new_pts = new_pts.transpose(0, 2, 1) # B x 3 x n_kp
    new_pts = (t @ new_pts) # B x 2 x n_kp
    return new_pts.transpose(0, 2, 1)
    
def get_hm(heatmaps_uv, joints_2D, int_mode=False, radius=16):
    n_kp, _ = joints_2D.shape
    for i in range(n_kp):
        conf = 1
        if int_mode:
            draw_umich_gaussian_int(heatmaps_uv[i], joints_2D[i], radius, k=conf)
        else:
            draw_umich_gaussian(heatmaps_uv[i], joints_2D[i], radius, k=conf)
    return heatmaps_uv

def draw_umich_gaussian(heatmap, center, radius, k=1):
    # import pdb; pdb.set_trace()
    diameter = 2 * radius + 1
    height, width = heatmap.shape[0:2]
    x, y = int(center[0]), int(center[1])
    # gaussian = gaussian2D((diameter, diameter), sigma=2)
    
    if x - radius >=0 and x + radius + 1 < width and y - radius >= 0 and y + radius + 1 < height:
        # res = [0, 0]  #
        # print(res)
        res = [center[0] -x , center[1] - y]
        gaussian = gaussian2D((diameter, diameter), sigma=2, res=res)
        # height, width = heatmap.shape[0:2]
          
        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)
        # import pdb; pdb.set_trace()
        masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
          np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def draw_umich_gaussian_int(heatmap, center, radius, k=1):
    # import pdb; pdb.set_trace()
    diameter = 2 * radius + 1
    height, width = heatmap.shape[0:2]
    x, y = math.floor(center[0]), math.floor(center[1])
    # gaussian = gaussian2D((diameter, diameter), sigma=2)
    
    if x - radius >=0 and x + radius + 1 < width and y - radius >= 0 and y + radius + 1 < height:
        res = [0, 0]  #
        # print(res)
        # res = [center[0] -x , center[1] - y]
        gaussian = gaussian2D((diameter, diameter), sigma=2, res=res)
        # height, width = heatmap.shape[0:2]
          
        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)
        # import pdb; pdb.set_trace()
        masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
          np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def gaussian2D(shape, sigma, res):
    res_x, res_y = res
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    # y, x = np.arange(-m, m + 1).reshape(-1, 1), np.arange(-n, n + 1).reshape(1, -1)
    h = np.exp(-((x-res_x)**2 + (y-res_y)**2) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def normalize_image(inp, mean=None, std=None):
    if mean is not None and std is not None:
        inp = np.clip((inp.astype(np.float64) / 255.), 0.0, 1.0)
        inp = (inp - mean) /std  
    return inp

















import numpy as np
import os
import json
import cv2

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
        inp = (inp.astype(np.float64) / 255.)
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

def affine_transform_pts(pts, t):
    n_kp, _ = pts.shape
    new_ones = np.ones(n_kp).reshape(n_kp, 1)
    new_pts = np.concatenate((pts, new_ones), axis=-1)
    new_pts = np.dot(t, new_pts.T) # 得到一个2 x n_kp的矩阵
    new_pts = new_pts.T
    
    return new_pts

def batch_affine_transform_pts(pts, t):
    # pts : B x n_kp x 2
    # t : B x 2 x 3
    b, n_kp, _ = pts.shape
    new_ones = np.ones((b, n_kp, 1))
    new_pts = np.concatenate((pts, new_ones), axis=-1)
    new_pts = new_pts.transpose(0, 2, 1) # B x 3 x n_kp
    new_pts = (t @ new_pts) # B x 2 x n_kp
    return new_pts.transpose(0, 2, 1)
    
def get_hm(heatmaps_uv, joints_2D, int_mode=False, radius=16):
    n_kp, _ = joints_2D.shape
    for i in range(n_kp):
        conf = 1
        if int_mode:
            draw_umich_gaussian_int(heatmaps_uv[i], joints_2D[i], radius, k=conf)
        else:
            draw_umich_gaussian(heatmaps_uv[i], joints_2D[i], radius, k=conf)
    return heatmaps_uv

def draw_umich_gaussian(heatmap, center, radius, k=1):
    # import pdb; pdb.set_trace()
    diameter = 2 * radius + 1
    height, width = heatmap.shape[0:2]
    x, y = int(center[0]), int(center[1])
    # gaussian = gaussian2D((diameter, diameter), sigma=2)
    
    if x - radius >=0 and x + radius + 1 < width and y - radius >= 0 and y + radius + 1 < height:
        # res = [0, 0]  #
        # print(res)
        res = [center[0] -x , center[1] - y]
        gaussian = gaussian2D((diameter, diameter), sigma=2, res=res)
        # height, width = heatmap.shape[0:2]
          
        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)
        # import pdb; pdb.set_trace()
        masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
          np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def draw_umich_gaussian_int(heatmap, center, radius, k=1):
    # import pdb; pdb.set_trace()
    diameter = 2 * radius + 1
    height, width = heatmap.shape[0:2]
    x, y = math.floor(center[0]), math.floor(center[1])
    # gaussian = gaussian2D((diameter, diameter), sigma=2)
    
    if x - radius >=0 and x + radius + 1 < width and y - radius >= 0 and y + radius + 1 < height:
        res = [0, 0]  #
        # print(res)
        # res = [center[0] -x , center[1] - y]
        gaussian = gaussian2D((diameter, diameter), sigma=2, res=res)
        # height, width = heatmap.shape[0:2]
          
        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)
        # import pdb; pdb.set_trace()
        masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
          np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def gaussian2D(shape, sigma, res):
    res_x, res_y = res
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    # y, x = np.arange(-m, m + 1).reshape(-1, 1), np.arange(-n, n + 1).reshape(1, -1)
    h = np.exp(-((x-res_x)**2 + (y-res_y)**2) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h



















