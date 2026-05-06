import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import copy
import torch
from models.graspnet.graspnet_baseline import GraspNetBaseLine
from utils import graspnet_config


# Modified from https://github.com/OCRTOC/OCRTOC_software_package/blob/master/ocrtoc_perception/src/ocrtoc_perception/perceptor.py
class Graspnet:
    def __init__(self):
        self.config = graspnet_config
        self.graspnet_baseline = GraspNetBaseLine(checkpoint_path = self.config['graspnet_checkpoint_path'])

    def compute_grasp_pose(self, full_pcd):

        points = np.asarray(full_pcd.points)

        grasp_pcd = copy.deepcopy(full_pcd)
        grasp_pcd.points = o3d.utility.Vector3dVector(-points)
        gg = self.graspnet_baseline.inference(grasp_pcd)
        gg.translations = -gg.translations
        gg.rotation_matrices = -gg.rotation_matrices
        gg.translations = gg.translations + gg.rotation_matrices[:, :, 0] * self.config['refine_approach_dist']

        gg = self.graspnet_baseline.collision_detection(gg, points)
        return gg

    def assign_grasp_pose(self, gg, object_pose):
        grasp_poses = []
        grasp_pose_set = []
        dist_thresh = self.config['dist_thresh']
        angle_thresh = self.config['angle_thresh']
        
        ts = gg.translations
        rs = gg.rotation_matrices
        depths = gg.depths
        scores = gg.scores

        ts = ts + rs[:,:,0] * (np.vstack((depths, depths, depths)).T)
        eelink_rs = np.zeros(shape=(len(rs), 3, 3), dtype=np.float32)
        eelink_rs[:,:,0] = rs[:,:,2]
        eelink_rs[:,:,1] = -rs[:,:,1]
        eelink_rs[:,:,2] = rs[:,:,0]

        object_position = object_pose[:3, 3]
        dists = np.linalg.norm(ts - object_position, axis=1)
        dist_mask = dists < dist_thresh
        angle_mask = (rs[:, 2, 0] < -np.cos(angle_thresh / 180.0 * np.pi))

        add_angle_mask = dist_mask & angle_mask

        if np.sum(add_angle_mask) < self.config['mask_thresh']:
            mask = dist_mask
            sorting_method = 'angle'
        else:
            mask = add_angle_mask
            sorting_method = 'score'

        i_scores = scores[mask]
        i_ts = ts[mask]
        i_eelink_rs = eelink_rs[mask]
        i_gg = gg[mask]

        remain_gg = []
        
        if np.sum(mask) < self.config['mask_thresh']:
            return [], None, []
        else:
            for i in range(len(i_gg)):
                remain_gg.append(i_gg[i].to_open3d_geometry())
                grasp_rotation_matrix = i_eelink_rs[i]
                if np.linalg.norm(np.cross(grasp_rotation_matrix[:,0], grasp_rotation_matrix[:,1]) - grasp_rotation_matrix[:,2]) > 0.1:
                    grasp_rotation_matrix[:,0] = -grasp_rotation_matrix[:,0]

                grasp_pose = np.zeros(7)
                grasp_pose[:3] = i_ts[i]
                r = R.from_matrix(grasp_rotation_matrix)
                grasp_pose[-4:] = r.as_quat()

                grasp_poses.append(grasp_pose)
                grasp_pose_set.append(grasp_pose)

        return grasp_pose_set, grasp_poses, remain_gg

    
    def grasp_detection(self, full_pcd, object_poses=None):
        '''
        Generate object 6d poses and grasping poses.
        Only geometry infomation is used in this implementation.

        There are mainly three steps.
        - Moving the camera to different predefined locations and capture RGBD images. Reconstruct the 3D scene.
        - Generating objects 6d poses by mainly icp matching.
        - Generating grasping poses by graspnet-baseline.

        Args:
            object_list(list): strings of object names.
            pose_method: string of the 6d pose estimation method, "icp" or "superglue".
        Returns:
            dict, dict: object 6d poses and grasp poses.
        '''

        # generate grasping poses in a scene 
        gg = self.compute_grasp_pose(full_pcd)
        gg.sort_by_score()
        gg = gg[:20]
        # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
        # o3d.visualization.draw_geometries([frame, full_pcd, *gg.to_open3d_geometry_list()])
        # visualization
        # grasp_pose_set, grasp_pose_dict, remain_gg = self.assign_grasp_pose(gg, object_poses)
        
        # o3d.visualization.draw_geometries([frame, full_pcd, *gg.to_open3d_geometry_list()])

        return gg


