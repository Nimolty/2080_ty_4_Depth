import numpy as np
import open3d as o3d
import os
import torch
import json
from tqdm import tqdm

def exists_or_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    else:
        return True

def save_view_point(pcd_list, filename, width=640, height=360):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height)
    for pcd in pcd_list:
        vis.add_geometry(pcd)
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()


def load_view_point(pcd_list, filename, width=640, height=360):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height)
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(filename)
    for pcd in pcd_list:
        vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    vis.destroy_window()

def visualize_dynamic_pts(sampling_tensor_lists, gt_pts_cam_tensor, final_pred_pts_cam_tensor, c2r_rot_tensor, c2r_trans_tensor, save_dir, cam_params_path,width=640, height=360):
    # sampling_lists : list, length of 1000steps for ddpm
    # gt_pts : N x 3 (N is the num of keypoints)
    # c2r_rot : 3 x 3
    # c2r_trans : 3 x 1 
    # all of them are tensors
    
    # construct visualizer
    vis=o3d.visualization.Visualizer()
    vis.create_window(width=width,height=height, visible=False)
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    opt = vis.get_render_option()
    print("opt :", opt)
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 1
    opt.show_coordinate_frame = True
    param=o3d.io.read_pinhole_camera_parameters(cam_params_path)
    ctr=vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(param)

  
    # construct gt_pts in robot space, pointcloud meshes and lines meshes
    gt_pts_rob_tensor = ((c2r_rot_tensor.T) @ (gt_pts_cam_tensor.T - c2r_trans_tensor)).T
    gt_pts_rob_np = gt_pts_rob_tensor.detach().cpu().numpy() # N x 3
    
    final_pred_pts_rob_tensor = ((c2r_rot_tensor.T) @ (final_pred_pts_cam_tensor.T - c2r_trans_tensor)).T
    final_pred_pts_rob_np = final_pred_pts_rob_tensor.detach().cpu().numpy() # N x 3
    
    # construct gt pointcloud meshes
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points= o3d.utility.Vector3dVector(gt_pts_rob_np)
    gt_pcd.paint_uniform_color([0.0, 1.0, 0.0])
    
    # construct gt lines meshes
    gt_line = o3d.geometry.LineSet()
    gt_line.points = o3d.utility.Vector3dVector(gt_pts_rob_np)
    line_idx = np.array([
                        [0, 1], [0, 2],
                        [1, 3], [3, 4],
                        [3, 5],
                        [5, 6],[6, 7], 
                        [6, 8],
                        [8, 9], [9, 10],
                        [9, 11], 
                        [11, 12], [12, 13],
                        ])
    gt_line.lines = o3d.utility.Vector2iVector(line_idx)    
    gt_line.paint_uniform_color([0.0, 1.0, 0.0])

    #
    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points= o3d.utility.Vector3dVector(final_pred_pts_rob_np)
    final_pcd.paint_uniform_color([0.0, 0.0, 1.0])
    final_line = o3d.geometry.LineSet()
    final_line.points = o3d.utility.Vector3dVector(final_pred_pts_rob_np)
    final_line.lines = o3d.utility.Vector2iVector(line_idx)    
    final_line.paint_uniform_color([0.0, 0.0, 1.0])

    # construct pred pointcloud meshes
    pred_pcd = o3d.geometry.PointCloud()
    pred_pcd.paint_uniform_color([1.0, 0.0, 0.0])
    
    # construct pred lines meshes
    pred_line = o3d.geometry.LineSet()
    pred_line.paint_uniform_color([1.0, 0.0, 0.0])
    
    vis.add_geometry(axis_pcd)
    vis.add_geometry(gt_line)
    vis.add_geometry(gt_pcd)
    vis.add_geometry(pred_line)
    vis.add_geometry(pred_pcd)
    vis.add_geometry(final_line)
    vis.add_geometry(final_pcd)
    
    for b in tqdm(range(len(sampling_tensor_lists))):
        pred_pts_cam_tensor = sampling_tensor_lists[b]
        #print("pred_pts_cam_tensor", pred_pts_cam_tensor)
        pred_pts_rob_tensor = ((c2r_rot_tensor.T) @ (pred_pts_cam_tensor.T - c2r_trans_tensor)).T
        pred_pts_rob_np = pred_pts_rob_tensor.detach().cpu().numpy() # N x 3
        
        pred_pcd.points = o3d.utility.Vector3dVector(pred_pts_rob_np)
        pred_line.points = pred_pcd.points
        pred_line.lines = o3d.utility.Vector2iVector(line_idx)  
        pred_pcd.paint_uniform_color([1.0, 0.0, 0.0])  
        pred_line.paint_uniform_color([1.0, 0.0, 0.0])
        
        vis.update_geometry(pred_line) 
        vis.update_geometry(pred_pcd)
        ctr.convert_from_pinhole_camera_parameters(param)
        
        vis.poll_events()
        vis.update_renderer()
        
        exists_or_mkdir(save_dir)
        vis.capture_screen_image(os.path.join(save_dir, f"{str(b).zfill(4)}.png"), False)
    
    vis.destroy_window()
        

if __name__ == "__main__":    
#    point_cloud_np = np.array([[0.0, 0.0, 0.0], 
#                               [1.0, 0.0, 0.0],
#                               [0.0, 1.0, 0.0],
#                               [0.0, 0.0, 1.0],
#                              ])
#    point_cloud = o3d.geometry.PointCloud()
#    point_cloud.points = o3d.utility.Vector3dVector(point_cloud_np)
#    
#    save_view_point(point_cloud, f"./viewpoint.json")
#    load_view_point(point_cloud, f"./viewpoint.json")
    width, height = 320, 180
    json_path = "/DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/4_kinect_front_1/000500.json"
    with open(json_path, 'r') as fd:
        json_data = json.load(fd)[0]
    json_joints_data = json_data["joints_3n_fixed_42"]
    json_keypoints_data = json_data["keypoints"]   
    
    joints_loc_wrt_cam_data = torch.from_numpy(np.array([json_joints_data[idx]["location_wrt_cam"] for idx in range(len(json_joints_data))]))
    c2r_rot_tensor = torch.from_numpy(np.array(json_keypoints_data[0]["R2C_mat"]))
    c2r_trans_tensor = torch.from_numpy(np.array(json_keypoints_data[0]["location_wrt_cam"])).reshape(3, 1)
    sampling_tensor_lists = [joints_loc_wrt_cam_data + 0.02 * torch.randn_like(joints_loc_wrt_cam_data) for i in range(40)]
    gt_pts_cam_tensor = joints_loc_wrt_cam_data
    
    point_cloud = o3d.geometry.PointCloud()
    point_cloud_np = (((c2r_rot_tensor.T) @ (joints_loc_wrt_cam_data .T - c2r_trans_tensor)).T).detach().cpu().numpy()
    point_cloud.points = o3d.utility.Vector3dVector(point_cloud_np)
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    save_view_point([point_cloud, axis_pcd], f"./viewpoint.json", width=width, height=height)
    load_view_point([point_cloud, axis_pcd], f"./viewpoint.json", width=width, height=height)
    
    
    refer_path = "/DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/4_kinect_front_1/000500.npy"
    save_dir = "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/depth_c2rp/diffusion_utils/diffusion_visual"
    cam_params_path = "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/depth_c2rp/diffusion_utils/viewpoint.json"
    
    visualize_dynamic_pts(sampling_tensor_lists, gt_pts_cam_tensor, c2r_rot_tensor, c2r_trans_tensor, refer_path, save_dir, cam_params_path,
                          width=width, height=height)

