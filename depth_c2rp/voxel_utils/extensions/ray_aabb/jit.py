from torch.utils.cpp_extension import load
ray_aabb = load(
    'ray_aabb', ["/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/depth_c2rp/voxel_utils/extensions/ray_aabb/ray_aabb_cuda.cpp", "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/depth_c2rp/voxel_utils/extensions/ray_aabb/ray_aabb_cuda_kernel.cu"], verbose=True)
# help(ray_aabb)
