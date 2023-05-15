import glob
import cv2
import numpy as np
import json
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1" 
import time
from tqdm import tqdm


if __name__ == "__main__":
#    root_path = "/DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test/depth/"
#    depth_path_list = glob.glob(os.path.join(root_path, "*.exr"))
#    json_path_list = glob.glob(os.path.join(root_path, "*.json"))
#    npy_path_list = glob.glob(os.path.join(root_path, "*.npy"))
#    
    root_path = "/DATA/disk1/hyperplane/Depth_C2RP/Data/Data_0201_test_syn/"
    depth_path_list = glob.glob(os.path.join(root_path, "*", "*simDepthImage.exr"))
    json_path_list = glob.glob(os.path.join(root_path, "*", "*.json"))
    npy_path_list = glob.glob(os.path.join(root_path, "*", "*.npy"))
    
    json_lst = []
    depth_lst = []
    npy_lst = []
    for idx, depth_path in enumerate(tqdm(depth_path_list)):
#        t1 = time.time()
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)[:, :, 0].astype(np.float32)
#        t2 = time.time()
#        depth_lst.append(t2 - t1)
#        print("depth t2 - t1", t2 - t1)
#        print(depth_path)
        np.save(depth_path.replace("exr", "npy"), depth_img)
#        
#    for idx, npy_path in enumerate(tqdm(npy_path_list)):
#        t1 = time.time()
#        depth_img = np.load(npy_path)
#        print("depth_img.shape", depth_img.shape)
#        t2 = time.time()
#        npy_lst.append(t2 - t1)
#        print("npy t2 - t1", t2 - t1)
#        print(depth_path)
#        np.save(depth_path.replace("exr", "npy"), depth_img)
    
#    for idx, json_path in enumerate(json_path_list):
#        t1 = time.time()
#        print(json_path)
#        json.load(open(json_path, 'r'))
#        t2 = time.time()
#        print("json t2 - t1", t2 - t1)
#        json_lst.append(t2 - t1)
    
    print("depth mean", np.mean(depth_lst))
    print("json mean", np.mean(json_lst))
    print("npy_lst", np.mean(npy_lst))