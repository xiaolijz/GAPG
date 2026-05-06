import math
import numpy as np
import pybullet as p
import cv2
import open3d as o3d
import re
from shapely.geometry import Point, Polygon,MultiPolygon
from scipy.ndimage import binary_fill_holes
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from env.constants import WORKSPACE_LIMITS, PIXEL_SIZE
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from typing import Union
from pytorch3d.ops import sample_farthest_points
from typing import Literal, Optional, Tuple, Union
try:
    from scipy.spatial import cKDTree
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

reconstruction_config = {
    'nb_neighbors': 50,
    'std_ratio': 2.0,
    'voxel_size': 0.0015,
    'icp_max_try': 5,
    'icp_max_iter': 2000,
    'translation_thresh': 3.95,
    'rotation_thresh': 0.02,
    'max_correspondence_distance': 0.02
}

graspnet_config = {
    'graspnet_checkpoint_path': 'models/graspnet/checkpoints/checkpoint-rs.tar',
    'refine_approach_dist': 0.01,
    'dist_thresh': 0.05,
    'angle_thresh': 10,
    'mask_thresh': 0.5
}

def get_pointcloud(depth, intrinsics):
    """Get 3D pointcloud from perspective depth image.
    Args:
        depth: HxW float array of perspective depth in meters.
        intrinsics: 3x3 float array of camera intrinsics matrix.
    Returns:
        points: HxWx3 float array of 3D points in camera coordinates.
    """
    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
    points = np.float32([px, py, depth]).transpose(1, 2, 0)

    return points

def get_mask_pointcloud(depth, intrinsics, seg, target_id):
    """

    """
    segm = np.array(seg, dtype=np.int32).reshape(depth.shape)
    object_mask = (segm == target_id)|(segm == 1)
    object_pcd_mask = (segm == target_id)

    depth_mask = depth.copy()
    depth_mask[~object_mask] = 0  

    depth_pcd = depth.copy()
    depth_pcd[~object_pcd_mask] = 0

    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
    points = np.float32([px, py, depth_mask]).transpose(1, 2, 0)
    points_pcd = np.float32([px, py, depth_pcd]).transpose(1, 2, 0)
    return points, points_pcd

def get_all_obj_mask_pointcloud(depth, intrinsics, seg, all_obj_id):
    """
    get all obj points for push sample
    """
    segm = np.array(seg, dtype=np.int32).reshape(depth.shape)
    all_points = []

    for id in all_obj_id:
        object_pcd_mask = (segm == id)

        depth_pcd = depth.copy()
        depth_pcd[~object_pcd_mask] = 0

        height, width = depth.shape
        xlin = np.linspace(0, width - 1, width)
        ylin = np.linspace(0, height - 1, height)
        px, py = np.meshgrid(xlin, ylin)
        px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
        py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])

        point_pcd = np.float32([px, py, depth_pcd]).transpose(1, 2, 0)
        all_points.append(point_pcd)
    return all_points
 
def transform_pointcloud(points, transform):
    """Apply rigid transformation to 3D pointcloud.
    Args:
        points: HxWx3 float array of 3D points in camera coordinates.
        transform: 4x4 float array representing a rigid transformation matrix.
    Returns:
        points: HxWx3 float array of transformed 3D points.
    """
    padding = ((0, 0), (0, 0), (0, 1))
    homogen_points = np.pad(points.copy(), padding, "constant", constant_values=1)
    for i in range(3):
        points[Ellipsis, i] = np.sum(transform[i, :] * homogen_points, axis=-1)
    return points

def process_pcds(pcds, reconstruction_config):
    trans = dict()
    pcd = pcds[0]
    pcd.estimate_normals()
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors = reconstruction_config['nb_neighbors'],
        std_ratio = reconstruction_config['std_ratio']
    )
    for i in range(1, len(pcds)):
        voxel_size = reconstruction_config['voxel_size']
        income_pcd, _ = pcds[i].remove_statistical_outlier(
            nb_neighbors = reconstruction_config['nb_neighbors'],
            std_ratio = reconstruction_config['std_ratio']
        )
        income_pcd.estimate_normals()
        income_pcd = income_pcd.voxel_down_sample(voxel_size)
        transok_flag = False
        for _ in range(reconstruction_config['icp_max_try']): # try 5 times max
            reg_p2p = o3d.pipelines.registration.registration_icp(
                income_pcd,
                pcd,
                reconstruction_config['max_correspondence_distance'],
                np.eye(4, dtype = np.float),
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(reconstruction_config['icp_max_iter'])
            )
            if (np.trace(reg_p2p.transformation) > reconstruction_config['translation_thresh']) \
                and (np.linalg.norm(reg_p2p.transformation[:3, 3]) < reconstruction_config['rotation_thresh']):
                # trace for transformation matrix should be larger than 3.5
                # translation should less than 0.05
                transok_flag = True
                break
        if not transok_flag:
            reg_p2p.transformation = np.eye(4, dtype = np.float32)
        income_pcd = income_pcd.transform(reg_p2p.transformation)
        trans[i] = reg_p2p.transformation
        pcd += income_pcd
        pcd = pcd.voxel_down_sample(voxel_size)
        pcd.estimate_normals()
    return trans, pcd

def process_pcds_test(pcds):
    points_state_list = []
    colors = []
    for pcd in pcds:
        points = np.asarray(pcd.points)
        color = np.asarray(pcd.colors)
        points_state_list.append(points)
        colors.append(color)

    points_state = np.vstack(points_state_list)
    colors_state = np.vstack(colors)
    points_pcd = o3d.geometry.PointCloud()
    points_pcd.points = o3d.utility.Vector3dVector(points_state)
    points_pcd.colors = o3d.utility.Vector3dVector(colors_state)

    return points_pcd

def process_all_pcds(all_config_pcd):
    obj_points_list = []

    for i in range(len(all_config_pcd[0])):
        obj_point = []
        for j in range(len(all_config_pcd)):
            pcd = all_config_pcd[j][i]
            points = np.asarray(pcd.points)

            obj_point.append(points)

        obj_point = np.vstack(obj_point)
        obj_points_list.append(obj_point)
    obj_pcds_list = []
    for i in range(len(obj_points_list)):
        points_pcd = o3d.geometry.PointCloud()
        points_pcd.points = o3d.utility.Vector3dVector(obj_points_list[i])
        obj_pcds_list.append(points_pcd)

    return obj_pcds_list

def get_fuse_pointcloud(env, obj_id, id=0):
    pcds = []
    configs = [env.oracle_cams[0], env.agent_cams[0], env.agent_cams[1], env.agent_cams[2]]
    # Capture near-orthographic RGB-D images and segmentation masks.
    for config in configs:
        color, depth, seg = env.render_camera(config)
        if id == 0:
            xyz, _ = get_mask_pointcloud(depth, config["intrinsics"], seg, obj_id)
        else:
            _, xyz = get_mask_pointcloud(depth, config["intrinsics"], seg, obj_id)
        # xyz = get_pointcloud(depth, config["intrinsics"])
        position = np.array(config["position"]).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(config["rotation"])
        rotation = np.array(rotation).reshape(3, 3)
        transform = np.eye(4)
        transform[:3, :] = np.hstack((rotation, position))
        points = transform_pointcloud(xyz, transform)
        # Filter out 3D points that are outside of the predefined bounds.
        ix = (points[Ellipsis, 0] >= env.bounds[0, 0]) & (points[Ellipsis, 0] < env.bounds[0, 1])
        iy = (points[Ellipsis, 1] >= env.bounds[1, 0]) & (points[Ellipsis, 1] < env.bounds[1, 1])
        iz = (points[Ellipsis, 2] >= env.bounds[2, 0]) & (points[Ellipsis, 2] < env.bounds[2, 1])
        valid = ix & iy & iz
        points = points[valid]
        colors = color[valid]

        iz = np.argsort(points[:, -1])
        points, colors = points[iz], colors[iz]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        pcd.voxel_down_sample(reconstruction_config['voxel_size'])
        # # visualization
        # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
        # o3d.visualization.draw_geometries([pcd, frame])
        # the first pcd is the one for start fusion
        pcds.append(pcd)

    # _, fuse_pcd = process_pcds(pcds, reconstruction_config)
    fuse_pcd = process_pcds_test(pcds)
    # visualization
    # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    # o3d.visualization.draw_geometries([fuse_pcd, frame])

    return fuse_pcd

def get_all_obj_pointcloud(env, obj_lis):
    all_config_pcds = []
    configs = [env.oracle_cams[0], env.agent_cams[0], env.agent_cams[1], env.agent_cams[2]]
    # Capture near-orthographic RGB-D images and segmentation masks.
    for config in configs:
        one_config_pcds = []
        color, depth, seg = env.render_camera(config)
        all_xyz = get_all_obj_mask_pointcloud(depth, config['intrinsics'], seg, obj_lis)
        # xyz = get_pointcloud(depth, config["intrinsics"])
        for xyz in all_xyz:
            position = np.array(config["position"]).reshape(3, 1)
            rotation = p.getMatrixFromQuaternion(config["rotation"])
            rotation = np.array(rotation).reshape(3, 3)
            transform = np.eye(4)
            transform[:3, :] = np.hstack((rotation, position))
            points = transform_pointcloud(xyz, transform)
            # Filter out 3D points that are outside of the predefined bounds.
            ix = (points[Ellipsis, 0] >= env.bounds[0, 0]) & (points[Ellipsis, 0] < env.bounds[0, 1])
            iy = (points[Ellipsis, 1] >= env.bounds[1, 0]) & (points[Ellipsis, 1] < env.bounds[1, 1])
            iz = (points[Ellipsis, 2] >= env.bounds[2, 0]) & (points[Ellipsis, 2] < env.bounds[2, 1])
            valid = ix & iy & iz
            points = points[valid]
            colors = color[valid]
            iz = np.argsort(points[:, -1])
            points, colors = points[iz], colors[iz]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
            pcd.voxel_down_sample(reconstruction_config['voxel_size'])
            # # visualization
            # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
            # o3d.visualization.draw_geometries([pcd, frame])
            # the first pcd is the one for start fusion
            one_config_pcds.append(pcd)
        all_config_pcds.append(one_config_pcds)
    # _, fuse_pcd = process_pcds(pcds, reconstruction_config)
    fuse_pcd = process_all_pcds(all_config_pcds)
    # visualization
    # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    # o3d.visualization.draw_geometries([fuse_pcd, frame])

    return fuse_pcd


def global_label_points(depth, intrinsics, seg, target_id):
    """
    assign labels to goal-obj and other obj
    goal-obj:[0,1,0] other:[0,0,1]
    """
    segm = np.array(seg, dtype=np.int32).reshape(depth.shape)
    goal_obj_mask = (segm == target_id)
    other_mask = (segm != target_id) & (segm != 1)  
    without_floor = (segm == 1)
    # crop floor
    depth_mask_global = depth.copy()
    depth_mask_global[without_floor] = 0
    # depth_mask_obj = depth.copy()
    # depth_mask_obj[~goal_obj_mask] = 0
    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
    
    points = np.float32([px, py, depth_mask_global]).transpose(1, 2, 0)
    # obj_points = np.float32([px, py, depth_mask_obj]).transpose(1, 2, 0)
    # add labels to global_pc
    labels = np.zeros((height, width, 2), dtype=np.float32)
    labels[goal_obj_mask] = [1.0, 0.0]  # goal-obj
    labels[other_mask] = [0.0, 1.0] # other-obj
    global_points_six = np.concatenate([points, labels], axis=-1) 

    return global_points_six

def get_global_pc(env):
    pcds = []
    segs = []
    configs = [env.oracle_cams[0], env.agent_cams[0], env.agent_cams[1], env.agent_cams[2]]
    for config in configs:
        color, depth, seg = env.render_camera(config)
        xyz = get_pointcloud(depth, config["intrinsics"])
        
        # xyz = get_pointcloud(depth, config["intrinsics"])
        position = np.array(config["position"]).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(config["rotation"])
        rotation = np.array(rotation).reshape(3, 3)
        transform = np.eye(4)
        transform[:3, :] = np.hstack((rotation, position))
        points = transform_pointcloud(xyz, transform)
        # Filter out 3D points that are outside of the predefined bounds.
        ix = (points[Ellipsis, 0] >= env.bounds[0, 0]) & (points[Ellipsis, 0] < env.bounds[0, 1])
        iy = (points[Ellipsis, 1] >= env.bounds[1, 0]) & (points[Ellipsis, 1] < env.bounds[1, 1])
        iz = (points[Ellipsis, 2] >= env.bounds[2, 0]) & (points[Ellipsis, 2] < env.bounds[2, 1])
        valid = ix & iy & iz
        points = points[valid]
        colors = color[valid]
        iz = np.argsort(points[:, -1])
        points, colors = points[iz], colors[iz]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        pcd.voxel_down_sample(reconstruction_config['voxel_size'])
        # # visualization
        # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
        # o3d.visualization.draw_geometries([pcd, frame])
        # the first pcd is the one for start fusion
        pcds.append(pcd)
        segs.append(seg)

    fuse_pcd = process_pcds_test(pcds)
    ply_global = furthest_point_sampling(fuse_pcd, n_samples=25000)
    # ply_global_for_eval = furthest_point_sampling(fuse_pcd, n_samples=18000)
    # pcd_global = o3d.geometry.PointCloud()
    # pcd_global.points = o3d.utility.Vector3dVector(ply_global)
    # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    # o3d.visualization.draw_geometries([pcd_global, frame])
    return ply_global, segs[0]

def get_global_label_pc(env, target_id):
    pcds = []
    segs = []
    configs = [env.oracle_cams[0], env.agent_cams[0], env.agent_cams[1], env.agent_cams[2]]
    # Capture near-orthographic RGB-D images and segmentation masks.
    for config in configs:
        color, depth, seg = env.render_camera(config)
        xyz_label = global_label_points(depth, config["intrinsics"], seg, target_id)  # HxWx5
        xyz_hw   = xyz_label[:, :, :3].astype(np.float64)   # HxWx3
        labels_hw = xyz_label[:, :, 3:].astype(np.float32)  # HxWx2
        H, W = depth.shape
        position = np.array(config["position"]).reshape(3, 1)
        rotation = np.array(p.getMatrixFromQuaternion(config["rotation"])).reshape(3, 3)
        T = np.eye(4); T[:3, :3] = rotation; T[:3, 3] = position[:, 0]

        points_hw = transform_pointcloud(xyz_hw, T)         # HxWx3
        points    = points_hw.reshape(-1, 3)                # N x 3
        labels    = labels_hw.reshape(-1, labels_hw.shape[-1])  # N x C 
        colors    = color.reshape(-1, 3).astype(np.float64)      # N x 3

        ix = (points[:, 0] >= env.bounds[0, 0]) & (points[:, 0] < env.bounds[0, 1])
        iy = (points[:, 1] >= env.bounds[1, 0]) & (points[:, 1] < env.bounds[1, 1])
        iz = (points[:, 2] >= env.bounds[2, 0]) & (points[:, 2] < env.bounds[2, 1])
        valid = ix & iy & iz

        points = points[valid]
        labels = labels[valid]
        colors = colors[valid]

        order = np.argsort(points[:, 2])
        points = points[order]
        labels = labels[order]
        colors = colors[order]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

        voxel = reconstruction_config['voxel_size']
        min_b = pcd.get_min_bound() - voxel
        max_b = pcd.get_max_bound() + voxel

        pcd_ds, _, traces = pcd.voxel_down_sample_and_trace(
            voxel_size=voxel,
            min_bound=min_b,
            max_bound=max_b,
            approximate_class=True
        )

        xyz_ds = np.asarray(pcd_ds.points).astype(np.float32)   # (M,3)
        keep_idx = np.array([np.asarray(idx, dtype=np.int64)[0] for idx in traces], dtype=np.int64)
        lab_ds = labels[keep_idx].astype(np.float32)           # (M,C)
        xyz_label_ds = np.concatenate([xyz_ds, lab_ds], axis=1).astype(np.float32)

        pcds.append(xyz_label_ds)
        segs.append(seg)
    points_state = np.vstack(pcds)
    ply_global = fps_xyz_label(points_state, n_samples=25000)
    return ply_global, segs[0]

def adjust_pose_z_axis_to_down(rot_matrix):

    x_axis = rot_matrix[:, 0]  # shape (3,)

    z_axis = np.array([0, 0, -1])

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    new_rot = np.stack([x_axis, y_axis, z_axis], axis=1)  

    new_quat = R.from_matrix(new_rot).as_quat()  # [x, y, z, w]
    return new_quat, new_rot

def furthest_point_sampling(points, colors=None, semantics=None, n_samples=4096):
    """
    points: [N, 3] tensor containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically &lt;&lt; N
    """
    # Convert points to PyTorch tensor if not already and move to GPU
    pcd_np = np.asarray(points.points)
    # pcd_np = points.cpu().numpy()
    points = torch.from_numpy(pcd_np).float().cuda()  # [N, 3]
    # points = points.to('cuda')
    if colors is not None:
        colors = torch.Tensor(colors).cuda()
    if semantics is not None:
        semantics = semantics.astype(np.int32)
        semantics = torch.Tensor(semantics).cuda()
    # Number of points
    num_points = points.size(0)  # N
    # Initialize an array for the sampled indices
    sample_inds = torch.zeros(n_samples, dtype=torch.long).cuda()  # [S]
    # Initialize distances to inf
    dists = torch.ones(num_points).cuda() * float("inf")  # [N]
    # Select the first point randomly
    selected = torch.randint(num_points, (1,), dtype=torch.long).cuda()  # [1]
    sample_inds[0] = selected
    # Iteratively select points for a maximum of n_samples
    for i in range(1, n_samples):
        # Find the distance to the last added point in selected
        last_added = sample_inds[i - 1]  # Scalar
        dist_to_last_added_point = torch.sum(
            (points[last_added] - points) ** 2, dim=-1
        )  # [N]
        # If closer, update distances
        dists = torch.min(dist_to_last_added_point, dists)  # [N]
        # Pick the one that has the largest distance to its nearest neighbor in the sampled set
        selected = torch.argmax(dists)  # Scalar
        sample_inds[i] = selected

    if colors is not None and semantics is not None:
        return (
            points[sample_inds].cpu().numpy(),
            colors[sample_inds].cpu().numpy(),
            semantics[sample_inds].cpu().numpy(),
        )  # [S, 3]
    elif colors is not None:
        return points[sample_inds].cpu().numpy(), colors[sample_inds].cpu().numpy()
    else:
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points[sample_inds].cpu().numpy())
        # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
        # o3d.visualization.draw_geometries([pcd, frame])
        return points[sample_inds].detach().cpu().numpy()

def fps_xyz_label(xyz_label: np.ndarray,
                  n_samples: int = 4096,
                  return_index: bool = False,
                  device: str = None,
                  start_idx: int = None,
                  seed: int = None):
    arr = np.asarray(xyz_label)
    assert arr.ndim == 2 and arr.shape[1] >= 4, "xyz_label should be (N, 3+C)"
    N, D = arr.shape
    C = D - 3
    S = min(n_samples, N)
    device = 'cuda'
    xyz = torch.from_numpy(arr[:, :3]).to(device=device, dtype=torch.float32)   
    lab = torch.from_numpy(arr[:, 3:]).to(device=device, dtype=torch.float32)  
    idx = torch.empty(S, dtype=torch.long, device=device)
    dists = torch.full((N,), float('inf'), device=device)
    if start_idx is None:
        if seed is not None:
            g = torch.Generator(device=device)
            g.manual_seed(int(seed))
            idx0 = torch.randint(N, (1,), generator=g, device=device)[0]
        else:
            idx0 = torch.randint(N, (1,), device=device)[0]
    else:
        idx0 = torch.tensor(start_idx, dtype=torch.long, device=device).clamp_(0, N-1)

    idx[0] = idx0
    for i in range(1, S):
        last = idx[i-1]
        dist2 = torch.sum((xyz - xyz[last])**2, dim=-1) 
        dists = torch.minimum(dists, dist2)
        idx[i] = torch.argmax(dists)
    sampled_xyz = xyz[idx]          
    sampled_lab = lab[idx]         
    sampled = torch.cat([sampled_xyz, sampled_lab], dim=1).cpu().numpy()  
    if return_index:
        return sampled, idx.cpu().numpy().astype(np.int64)
    else:
        return sampled

def furthest_point_sampling_nocuda(points, colors=None, semantics=None, n_samples=4096,start_idx=0):
    """
    points: [N, 3] tensor containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically &lt;&lt; N
    """
    if colors is not None:
        colors = torch.Tensor(colors).cuda()
    if semantics is not None:
        semantics = semantics.astype(np.int32)
        semantics = torch.Tensor(semantics).cuda()

    # Number of points
    num_points = points.shape[0] # N

    # Initialize an array for the sampled indices
    sample_inds = torch.zeros(n_samples, dtype=torch.long) # [S]

    # Initialize distances to inf
    dists = torch.ones(num_points) * float("inf")  # [N]

    # Select the first point randomly
    # selected = torch.randint(num_points, (1,), dtype=torch.long)  # [1]
    selected = torch.tensor([start_idx], dtype=torch.long, device=points.device)
    sample_inds[0] = selected

    # Iteratively select points for a maximum of n_samples
    for i in range(1, n_samples):
        # Find the distance to the last added point in selected
        last_added = sample_inds[i - 1]  # Scalar
        dist_to_last_added_point = torch.sum(
            (points[last_added] - points) ** 2, dim=-1
        )  # [N]

        # If closer, update distances
        dists = torch.min(dist_to_last_added_point, dists)  # [N]

        # Pick the one that has the largest distance to its nearest neighbor in the sampled set
        selected = torch.argmax(dists)  # Scalar
        sample_inds[i] = selected

    if colors is not None and semantics is not None:
        return (
            points[sample_inds].cpu().numpy(),
            colors[sample_inds].cpu().numpy(),
            semantics[sample_inds].cpu().numpy(),
        )  # [S, 3]
    elif colors is not None:
        return points[sample_inds].cpu().numpy(), colors[sample_inds].cpu().numpy()
    else:
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points[sample_inds].cpu().numpy())
        # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
        # o3d.visualization.draw_geometries([pcd, frame])
        return points[sample_inds]

def write_ply(points, filename):
    """
    save 3D-points and colors into ply file.
    points: [N, 3] (X, Y, Z)
    filename: output filename
    """
    # combine vertices and colors
    vertices = np.array(
        [tuple(point) for point in points],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")],
    )

    el = PlyElement.describe(vertices, "vertex")

    # save PLY file
    PlyData([el], text=True).write(filename)

def is_in_workplace(env,obj_num):
    is_in_workplace = True

    pos, _, _ = env.obj_info(obj_num)
    if pos[0] < WORKSPACE_LIMITS[0][0] or pos[0] > WORKSPACE_LIMITS[0][1] \
        or pos[1] < WORKSPACE_LIMITS[1][0] or pos[1] > WORKSPACE_LIMITS[1][1]:
        is_in_workplace = False
        print(f"\033[031m Target objects {obj_num} are not in the scene!\033[0m")
  
    return is_in_workplace

def grasp_pcd():
    finger1 = o3d.geometry.TriangleMesh.create_box(width=0.022, height=0.015, depth=0.05)
    finger1.translate([-0.011, -0.0575 , -0.05])

    finger2 = o3d.geometry.TriangleMesh.create_box(width=0.022, height=0.015, depth=0.05)
    finger2.translate([-0.011, 0.0425 , -0.05])

    finger3 = o3d.geometry.TriangleMesh.create_box(width=0.022, height=0.115, depth=0.001)
    finger3.translate([-0.011, -0.0575 , -0.05])
    # finger3 = o3d.geometry.TriangleMesh.create_box(width=0.022, height=0.115, depth=0.015)
    # finger3.translate([-0.011, -0.0575 , -0.065])
    gripper_mesh = finger1 + finger2 + finger3
    gripper_pcd = gripper_mesh.sample_points_poisson_disk(200) 
    gripper_points = torch.from_numpy(np.asarray(gripper_pcd.points)).float()

    return gripper_points, gripper_pcd

def push_gripper_pcd(z, n_target, oversample=2000, seed=55926):
    """
    gripper:[1,0,0]
    """
    finger1 = o3d.geometry.TriangleMesh.create_box(width=0.044, height=0.03, depth=0.05)
    finger1.translate([0.5 - 0.022, 0 - 0.015, z])
    # finger2 = o3d.geometry.TriangleMesh.create_box(width=0.022, height=0.015, depth=0.05)
    # finger2.translate([0.5, -0.015, z])
    gripper_mesh = finger1 
    pts_dense = _area_weighted_sample_on_mesh(gripper_mesh, n_samples=oversample, seed=seed)  

    pts_final = furthest_point_sampling_det(pts_dense, n_samples=n_target, start_idx=0)  # [n_target,3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_final.detach().cpu().numpy())

    return pts_final.float(), pcd

def TransformPCD2EndLink(point_cloud_base,pose):
    """
        Parameters:
        - point_cloud_base: (N, 3) numpy array
        - T_base_to_gripper: (4, 4) numpy array

        Returns:
        - point_cloud_gripper: (N, 3) numpy array
    """
    assert point_cloud_base.shape[1] == 3
    assert pose.shape[0] == 7

    device = point_cloud_base.device
    dtype = point_cloud_base.dtype

    position = pose[:3]  
    quat = pose[3:]      
    qx, qy, qz, qw = quat
    R_mat = torch.tensor([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ], dtype=dtype, device=device)  

    T = torch.eye(4, dtype=dtype, device=device)
    T[:3, :3] = R_mat
    T[:3, 3] = position
    T_inv = torch.linalg.inv(T)  # [4, 4]
    N = point_cloud_base.shape[0]
    ones = torch.ones((N, 1), dtype=dtype, device=device)
    points_homo = torch.cat([point_cloud_base, ones], dim=1)  # [N, 4]
    points_transformed = (T_inv @ points_homo.T).T  # [N, 4]
    return points_transformed[:, :3]

def Transform_Push2Fixed_point(global_pc, obj_pc, fixed_point, push_action):
    """
    All push actions must be normalized to a fixed reference point. 
    This ensures consistent left-to-right movement by the robot, which simplifies the learning process.
    """
    # push_pose = np.eye(4)
    # push_pose[:3,3] = push_action[:3]
    # push_pose[:3,:3] = R.from_quat(push_action[3:]).as_matrix()
    push_pose = torch.from_numpy(push_pose).float()
    fixed_pose = torch.eye(4)
    z = push_action[2]
    z = z.unsqueeze(-1)
    fixed_pose[:3,3] = torch.cat([fixed_point, z],dim=-1)
    fixed_pose[:3,:3] = torch.tensor([[0,-1,0],
                                      [-1,0,0],
                                      [0,0,-1]],dtype=float)
    T_2fixed =fixed_pose @ torch.linalg.inv(push_pose)
    obj_pc = torch.cat([obj_pc, torch.ones(obj_pc.shape[0], 1)],dim=1)
    global_pc = torch.cat([global_pc, torch.ones(global_pc.shape[0], 1)],dim=1)
    global_pc = (T_2fixed @ global_pc.T).T # NX4   
    obj_pc = (T_2fixed @ obj_pc.T).T # NX4 

    return global_pc[:,:3], obj_pc[:, :3]

def extend_obb_single_dir_along_global_z(pcd: o3d.geometry.PointCloud, factor: float = 10.0):

    assert factor >= 1.0, "factor should >= 1.0"

    obb = pcd.get_oriented_bounding_box()

    R = obb.R.copy()                
    extent = obb.extent.copy()      
    center = obb.center.copy()

    z = np.array([0.0, 0.0, 1.0])   
    dots = np.array([np.dot(R[:, i], z) for i in range(3)])     
    k = int(np.argmax(np.abs(dots)))                             

    old_len = extent[k]
    new_len = factor * old_len
    delta = new_len - old_len       

    if dots[k] > 0:
        dir_vec = -R[:, k]
    else:
        dir_vec =  R[:, k]


    center = center + 0.5 * delta * dir_vec
    extent[k] = new_len


    new_obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
    new_obb.color = (1, 0, 0)  
    obb.color = (0, 0, 1)      

    return obb, new_obb

def fuse_state_torch_v2(global_points: torch.Tensor,
                     gripper_pcd: torch.Tensor,
                     threshold: float = 0.0065,
                     ):
    device = global_points.device
    dtype = global_points.dtype

    gripper_point = gripper_pcd.detach().cpu().numpy()
    gripper_pc = o3d.geometry.PointCloud()
    gripper_pc.points = o3d.utility.Vector3dVector(gripper_point)

    # gripper_obb = gripper_pc.get_oriented_bounding_box()
    _,gripper_obb = extend_obb_single_dir_along_global_z(gripper_pc)

    global_points = global_points.detach().cpu().numpy()
    global_pcd = o3d.geometry.PointCloud()
    global_pcd.points = o3d.utility.Vector3dVector(global_points)
    points_in_gripper = global_pcd.crop(gripper_obb)
    points_in_grippers = torch.from_numpy(np.asarray(points_in_gripper.points)).to(device=device, dtype=dtype)
    fuse_points = points_in_grippers

    return fuse_points

def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]

def pc_normalize(pc: torch.Tensor):
    centroid = torch.mean(pc, dim=1)          
    pc = pc - centroid                         
    m = torch.max(torch.sqrt(torch.sum(pc**2, dim=2)))  
    pc = pc / m                                
    return pc

def pc_normalize_grasp(pc: torch.Tensor):
    centroid = torch.mean(pc, dim=0)          
    pc = pc - centroid                         
    m = torch.max(torch.sqrt(torch.sum(pc**2, dim=1)))  
    pc = pc / m                               
    return pc, centroid, m

def transform_points_to_camera(points_world, T_cam_base):
    num_points = points_world.shape[0]
    homo_points = np.hstack((points_world, np.ones((num_points, 1))))  # Nx4
    points_cam = (T_cam_base @ homo_points.T).T[:, :3]  # Nx3
    return points_cam

def project_points_to_image(points_cam, fx, fy, cx, cy):
    X, Y, Z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
    Z[Z <= 0] = 1e-6  
    u = (X * fx / Z + cx).astype(int)
    v = (Y * fy / Z + cy).astype(int)
    return u, v

def dilate_masks(masks, kernel_size=3, iterations=1):
    H, W = masks.shape
    object_ids = np.unique(masks)
    object_ids = object_ids[object_ids != 0] 
    dilated_mask = np.zeros_like(masks)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    for obj_id in object_ids:
        binary_mask = (masks == obj_id).astype(np.uint8)
        dilated = cv2.dilate(binary_mask, kernel, iterations=iterations)
        dilated_mask[dilated > 0] = obj_id

    return dilated_mask

def segment_pointcloud_by_mask(points, masks):
    masks = dilate_masks(masks)
    intrinsics = np.array([[630000.0, 0, 320], [0, 630000.0, 240], [0, 0, 1]])  
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    position = np.array([0.5, 0, 1000.0]) 
    rotation = p.getQuaternionFromEuler((0, np.pi, -np.pi / 2))
    rot_matrix = np.array(p.getMatrixFromQuaternion(rotation)).reshape(3, 3)

    T_base_to_cam = np.eye(4)
    T_base_to_cam[:3, :3] = rot_matrix
    T_base_to_cam[:3, 3] = position
    T_cam_base = np.linalg.inv(T_base_to_cam)

    # points = np.asarray(scene_pcd.points)
    H, W = masks.shape

    points_cam = transform_points_to_camera(points, T_cam_base)
    u, v = project_points_to_image(points_cam, fx, fy, cx, cy)

    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u_valid = u[valid]
    v_valid = v[valid]
    points_valid = points[valid]
    labels = masks[v_valid, u_valid]

    object_ids = np.unique(labels)
    object_ids = object_ids[object_ids > 0]

    object_pcds = []
    object_masks = []
    for obj_id in object_ids[1:]:
        idx = np.where(labels == obj_id)[0]
        obj_points = points_valid[idx]

        obj_pcd = o3d.geometry.PointCloud()
        obj_pcd.points = o3d.utility.Vector3dVector(obj_points)
        object_pcds.append(obj_pcd)

        mask = np.zeros((H,W),dtype=bool)
        mask[v_valid[idx],u_valid[idx]] = True
        object_masks.append(mask)

    return object_pcds,object_masks

def sample_surface_points(object_pcd,expand=0.016,step=0.03):
    points = np.asarray(object_pcd.points)
    aabb = object_pcd.get_axis_aligned_bounding_box()
    z_mean = (aabb.get_max_bound()[2] + aabb.get_min_bound()[2]) / 2
    xy = points[:, :2]
    resolution = 0.001
    xy_min = xy.min(axis=0)
    xy_min = xy.min(axis=0)
    xy_max = xy.max(axis=0)
    pad = 10
    img_size = np.ceil((xy_max - xy_min) / resolution).astype(int) + 2*pad
    img = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
    indices = ((xy - xy_min) / resolution).astype(int) + pad
    img[indices[:, 1], indices[:, 0]] = 255 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)
    img_filled = binary_fill_holes(closed>0).astype(np.uint8) * 255
    dilated_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(img_filled, dilated_kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise ValueError("No contour found from projected point cloud.")
    img_color = np.zeros((img_filled.shape[0],img_filled.shape[1],3),dtype=np.uint8)
    img_color[img_filled > 0] = [0,0,255]
    max_contour = max(contours, key=cv2.contourArea)
    # cv2.drawContours(img_color,max_contour,-1,(255,0,0),1)
    # cv2.imshow('Contours',img_color)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    contour_pts_image = max_contour[:, 0, :].astype(np.float32) 
    contour_pts_image -= pad
    contour_pts_world = contour_pts_image * resolution + xy_min
    poly = Polygon(contour_pts_world)
    offset_polygon = poly.buffer(expand)
    if isinstance(offset_polygon,MultiPolygon):
        offset_polygon = max(offset_polygon.geoms, key = lambda p: p.area)
    boundary_coords = np.array(offset_polygon.exterior.coords[:-1])  
    push_xy = interpolate_polygon_edges_with_step(boundary_coords,step)

    [0.276, 0.724], [-0.224, 0.224]
    sampled_points = []
    for uv_pt in push_xy:
        if 0.276 < uv_pt[0] < 0.724 and -0.224 < uv_pt[1] < 0.224:
            sampled_points.append([uv_pt[0], uv_pt[1], z_mean])

    return np.array(sampled_points)

def interpolate_polygon_edges_with_step(hull_pts, step=0.005):
    hull_pts = np.asarray(hull_pts, dtype=np.float32)
    n = len(hull_pts)
    if n < 2:
        return hull_pts.copy()

    edges = hull_pts[(np.arange(n) + 1) % n] - hull_pts 
    edge_lengths = np.linalg.norm(edges, axis=1)
    total_length = np.sum(edge_lengths)

    if total_length < 1e-6:
        return hull_pts[:1] 

    cumulative_lengths = np.cumsum(edge_lengths)
    num_samples = max(int(np.floor(total_length / step)), 1)
    sample_distances = np.linspace(0, total_length, num_samples, endpoint=False)

    sampled_pts = []
    edge_idx = 0
    curr_edge_start = hull_pts[0]
    curr_edge_vec = edges[0]
    curr_edge_len = edge_lengths[0]
    curr_cum_len = 0.0

    for d in sample_distances:

        while d >= cumulative_lengths[edge_idx]:
            curr_cum_len = cumulative_lengths[edge_idx]
            edge_idx = (edge_idx + 1) % n
            curr_edge_start = hull_pts[edge_idx]
            curr_edge_vec = edges[edge_idx]
            curr_edge_len = edge_lengths[edge_idx]

        t = (d - curr_cum_len) / curr_edge_len 
        pt = curr_edge_start + curr_edge_vec * t
        sampled_pts.append(pt)

    return np.array(sampled_pts)

def remove_points_near_other_cloud(pcd_A, pcd_B, radius):
    A_points = np.asarray(pcd_A.points)
    B_points = np.asarray(pcd_B.points)

    B_tree = cKDTree(B_points[:, :2])

    keep_mask = []
    for i in range(len(A_points)):
        a_xy = A_points[i, :2]
        a_z = A_points[i, 2]
        idxs = B_tree.query_ball_point(a_xy, r=radius)

        keep = True
        for j in idxs:
            if B_points[j, 2] > a_z:
                keep = False
                break
        keep_mask.append(keep)

    keep_mask = np.array(keep_mask)
    filtered_pcd_A = o3d.geometry.PointCloud()
    filtered_pcd_A.points = o3d.utility.Vector3dVector(A_points[keep_mask])

    return filtered_pcd_A

def compute_pose_dict(pcd_a,pcd_b):
    def compute_centroid(pcd):
        points = np.asarray(pcd.points)
        centroid = np.mean(points, axis=0)
        return centroid
    def compute_pose(point, centroid_b):
        direction = centroid_b - point
        direction_xy = direction[:2]
        direction_xy /= np.linalg.norm(direction_xy)  
        x_axis = np.array([direction_xy[0], direction_xy[1], 0])
        z_axis = np.array([0, 0, -1])  
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
        pose = np.eye(4)
        pose[:3, :3] = rotation_matrix
        pose[:3, 3] = point

        return pose

    centroid_b = compute_centroid(pcd_b)

    poses_dict = []

    for idx, point in enumerate(np.asarray(pcd_a.points)):
        pose = compute_pose(point, centroid_b)
        poses_dict.append(pose)

    return poses_dict

def get_push_pose(object_pcd, pcd1):

    sample_points = sample_surface_points(object_pcd)
    if len(sample_points) == 0:
       poses_dict = []
    else:
        sampled_pcd = o3d.geometry.PointCloud()
        sampled_pcd.points = o3d.utility.Vector3dVector(sample_points)
        filter_pcd = remove_points_near_other_cloud(sampled_pcd, pcd1, radius=0.015)
        # filter_pcd = remove_points_near_other_cloud(sampled_pcd, pcd1, radius=0.001)
        # poses_dict = compute_pose_dict(filter_pcd, object_pcd)
        poses_dict = compute_pose_dict(filter_pcd, object_pcd)
    return poses_dict


def sample_push_action(points,object_pcds):
    
    minZ = np.min(points[:, 2])
    indices = np.where(points[:, 2] >= minZ + 0.005)[0]
    new_points = points[indices]
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(new_points)
    # object_pcds,_ = segment_pointcloud_by_mask(points,masks)
    poses_dicts = []
    for i, object_pcd in enumerate(object_pcds):
        # o3d.visualization.draw_geometries([object_pcd])
        poses_dict = get_push_pose(object_pcd, pcd1)
        if len(poses_dict) == 0:
            continue
        poses_dicts.append(poses_dict)
    # secen_pcd = o3d.geometry.PointCloud()
    # secen_pcd.points = o3d.utility.Vector3dVector(points)
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(pcd1)
    # vis.add_geometry(secen_pcd)
    # for poses_dict in poses_dicts:  
    #     for pose in poses_dict:     
    #         coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.025)
    #         coordinate_frame.transform(pose)
    #         vis.add_geometry(coordinate_frame)

    # vis.run()
    # vis.destroy_window()
    return np.vstack(poses_dicts)

def transform_matrix2quat(push_actions):

    push_actions_sac = []
    for i in range(len(push_actions)):
        action = push_actions[i]
        position = action[:3, 3]
        rotation = action[:3, :3]
        r = R.from_matrix(rotation)
        quat = r.as_quat()
        t = np.hstack((position, quat))
        push_actions_sac.append(t)
    return np.vstack(push_actions_sac)

def _area_weighted_sample_on_mesh(mesh: o3d.geometry.TriangleMesh, n_samples: int, seed: int = 42) -> torch.Tensor:
    V = np.asarray(mesh.vertices)        
    F = np.asarray(mesh.triangles)       
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]
    tri_areas = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1) * 0.5
    area_sum = tri_areas.sum()
    if area_sum <= 0:
        raise ValueError("Mesh has zero total area.")

    probs = tri_areas / area_sum

    rng = np.random.RandomState(seed)

    face_idx = rng.choice(len(F), size=n_samples, replace=True, p=probs) 
    f0 = V[F[face_idx, 0]]
    f1 = V[F[face_idx, 1]]
    f2 = V[F[face_idx, 2]]
    u = rng.rand(n_samples, 1)
    v = rng.rand(n_samples, 1)
    su = np.sqrt(u)
    w0 = 1.0 - su
    w1 = su * (1.0 - v)
    w2 = su * v

    pts = (w0 * f0) + (w1 * f1) + (w2 * f2)  
    return torch.from_numpy(pts.astype(np.float32))  

def furthest_point_sampling_det(points: torch.Tensor, n_samples: int, start_idx: int = 0) -> torch.Tensor:
    if not torch.is_tensor(points):
        points = torch.tensor(points, dtype=torch.float32)
    device = points.device
    points = points.to(device=device, dtype=torch.float32)
    N = points.shape[0]
    n_samples = min(n_samples, N)
    sample_inds = torch.empty(n_samples, dtype=torch.long, device=device)
    dists = torch.full((N,), float("inf"), device=device)
    selected = torch.tensor([start_idx], dtype=torch.long, device=device)
    sample_inds[0] = selected
    for i in range(1, n_samples):
        last = sample_inds[i - 1]
        dist_to_last = torch.sum((points - points[last]) ** 2, dim=-1)
        dists = torch.minimum(dists, dist_to_last)
        selected = torch.argmax(dists)  
        sample_inds[i] = selected

    return points[sample_inds]

def grasp_pcd_bluenoise_like(n_target: int=500, oversample: int=5000, seed: int=42):
    finger1 = o3d.geometry.TriangleMesh.create_box(width=0.022, height=0.015, depth=0.05)
    finger1.translate([-0.011, -0.0575, -0.05])
    finger2 = o3d.geometry.TriangleMesh.create_box(width=0.022, height=0.015, depth=0.05)
    finger2.translate([-0.011, 0.0425, -0.05])
    finger3 = o3d.geometry.TriangleMesh.create_box(width=0.022, height=0.115, depth=0.015)
    finger3.translate([-0.011, -0.0575, -0.065])
    # finger3 = o3d.geometry.TriangleMesh.create_box(width=0.022, height=0.115, depth=0.001)
    # finger3.translate([-0.011, -0.0575 , -0.05])
    gripper_mesh = finger1 + finger2 + finger3

    pts_dense = _area_weighted_sample_on_mesh(gripper_mesh, n_samples=oversample, seed=seed)  # [M,3] CPU

    pts_final = furthest_point_sampling_det(pts_dense, n_samples=n_target, start_idx=0)  # [n_target,3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_final.detach().cpu().numpy())

    return pts_final.float(), pcd

def furthest_point_sampling_onehot_p3d(points, colors=None, semantics=None, n_samples=4096, start_idx=0):
    assert isinstance(points, torch.Tensor), "points should be torch.Tensor"
    device = points.device
    N = points.shape[0]
    K = min(int(n_samples), int(N))
    xyz = points[:, :3].to(dtype=torch.float32, device=device).contiguous().unsqueeze(0)  
    if start_idx is not None:
        if not (0 <= start_idx < N):
            raise ValueError(f"start_idx out of range")
        perm = torch.arange(N, device=device)
        if start_idx != 0:
            perm0 = perm[0].clone()
            perm[0] = perm[start_idx]
            perm[start_idx] = perm0
        xyz_perm = xyz[:, perm, :]  
        _, idx_perm = sample_farthest_points(xyz_perm, K=K, random_start_point=False)
        sel = perm[idx_perm[0]]  
    else:
        _, idx = sample_farthest_points(xyz, K=K, random_start_point=False)
        sel = idx[0]  
    out_points = points[sel]  
    out_colors = None
    out_semantics = None
    if colors is not None:
        out_colors = torch.as_tensor(colors, dtype=torch.float32, device=device)[sel]
    if semantics is not None:
        sem = torch.as_tensor(semantics, device=device)
        if sem.dtype != torch.long:
            sem = sem.to(torch.long)
        out_semantics = sem[sel]
    if out_colors is not None and out_semantics is not None:
        return out_points.cpu().numpy(), out_colors.cpu().numpy(), out_semantics.cpu().numpy()
    elif out_colors is not None:
        return out_points.cpu().numpy(), out_colors.cpu().numpy()
    else:
        return out_points

def Transform_Push2Fixed_point_onehot(global_points_onehot: torch.Tensor,
                                  fixed_point: torch.Tensor,
                                  push_action: torch.Tensor) -> torch.Tensor:


    dev  = 'cuda'
    dtype = global_points_onehot.dtype

    fixed_point = fixed_point.to(device=dev, dtype=dtype)      
    push_action = push_action.to(device=dev, dtype=dtype)      

    xyz   = global_points_onehot[:, :3]                       
    roles = global_points_onehot[:, 3:]                        

    t = push_action[:3]                                        
    q = push_action[3:7].unsqueeze(0)                          
    R_push = _quat_to_rotmat_torch(q).squeeze(0)               

    push_pose = torch.eye(4, device=dev, dtype=dtype)
    push_pose[:3, :3] = R_push
    push_pose[:3,  3] = t

    tz = push_action[2]   
    tz = tz.unsqueeze(0)                                   
    trans_fixed = torch.cat([fixed_point, tz], dim=-1)        

    R_fixed = torch.tensor([[0., -1.,  0.],
                            [-1.,  0.,  0.],
                            [0.,   0., -1.]], device=dev, dtype=dtype)
    fixed_pose = torch.eye(4, device=dev, dtype=dtype)
    fixed_pose[:3, :3] = R_fixed
    fixed_pose[:3,  3] = trans_fixed

    T_2fixed = fixed_pose @ torch.linalg.inv(push_pose)        

    N = xyz.shape[0]
    ones = torch.ones((N, 1), device=dev, dtype=dtype)
    xyz_h = torch.cat([xyz, ones], dim=1)                      
    xyz_tf = (T_2fixed @ xyz_h.T).T[:, :3]                     

    global_points_onehot_ee = torch.cat([xyz_tf, roles], dim=1)  

    return global_points_onehot_ee

def pc_normalize_grasp_onehot(pc: torch.Tensor):
    if not torch.is_tensor(pc):
        pc = torch.as_tensor(pc, dtype=torch.float32)
    if pc.ndim != 2 or pc.shape[1] < 3:
        raise ValueError(f"expected [N,>=3]，received {tuple(pc.shape)}")

    device, dtype = pc.device, pc.dtype
    N, C = pc.shape
    xyz   = pc[:, :3]                 
    extra = pc[:, 3:] if C > 3 else None  
    centroid = xyz.mean(dim=0)      
    xyz_c    = xyz - centroid         
    m = torch.linalg.norm(xyz_c, ord=2, dim=1).max()  
    eps = torch.tensor(1e-12, device=device, dtype=dtype)
    scale = torch.where(m > eps, m, torch.ones((), device=device, dtype=dtype))
    xyz_n = xyz_c / scale            
    pc_norm = torch.cat([xyz_n, extra], dim=1) if extra is not None else xyz_n
    return pc_norm, centroid, m

def _quat_to_rotmat_torch(q: torch.Tensor) -> torch.Tensor:
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-12)
    qx, qy, qz, qw = q.unbind(-1)
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz

    r00 = 1 - 2*(yy + zz)
    r01 =     2*(xy - wz)
    r02 =     2*(xz + wy)

    r10 =     2*(xy + wz)
    r11 = 1 - 2*(xx + zz)
    r12 =     2*(yz - wx)

    r20 =     2*(xz - wy)
    r21 =     2*(yz + wx)
    r22 = 1 - 2*(xx + yy)

    R = torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1)
    ], dim=-2)
    return R

def any_point_in_expanded_obb(
    A: Union[o3d.geometry.PointCloud, np.ndarray],
    B: Union[o3d.geometry.PointCloud, np.ndarray],
    expand_by: float = 0.5
) -> bool:
    obb = A.get_oriented_bounding_box()
    scale = 1.0 + float(expand_by)
    if scale <= 0:
        raise ValueError("scale is too small")
    obb_expanded = o3d.geometry.OrientedBoundingBox(obb.center, obb.R, obb.extent.copy())
    obb_expanded.scale(scale, obb_expanded.center)  

    ptsB = np.asarray(B.points)
    if ptsB.shape[0] == 0:
        return False
    ptsB_open3d = o3d.utility.Vector3dVector(ptsB)
    inside_idx = obb_expanded.get_point_indices_within_bounding_box(ptsB_open3d)
    return len(inside_idx) > 0