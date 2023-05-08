from torch.utils.cpp_extension import load
pcl_aabb = load(
    'pcl_aabb', ["/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/depth_c2rp/voxel_utils/extensions/pcl_aabb/pcl_aabb_cuda.cpp", "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/depth_c2rp/voxel_utils/extensions/pcl_aabb/pcl_aabb_cuda_kernel.cu"], verbose=True)
# help(pcl_aabb)
