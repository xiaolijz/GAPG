import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.push_networks import Space_Push_Fusion,Push_model
import utils  
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import random
from env.constants import WORKSPACE_LIMITS as workspace_limits
import time
from pytorch3d.ops import sample_farthest_points
# ---------- Dataset ----------
class PushEval():
    def __init__(self, model_dir, seed, device):
        # ---------- Arg Parser ----------
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        self.device = device
        self.model_dir = model_dir
        self.global_sample_count = 1024
        self.model = Push_model(additional_channel=3).to(device)
        self.checkpoint = torch.load(self.model_dir, map_location='cuda')  
        self.model.load_state_dict(self.checkpoint['model'])
        self.fixed_point = torch.tensor([(workspace_limits[0][0] + workspace_limits[0][1]) / 2, (workspace_limits[1][0] + workspace_limits[1][1]) / 2],dtype=float, device='cuda')
    def evalueate_push_actions(self, global_points_2onehot, push_actions):
        # preds = []
        push_actions = torch.from_numpy(push_actions).float().to('cuda')
        best_pre = None
        uniform_fuse_points = []
        K_target = self.global_sample_count
        self.model.eval()
        with torch.no_grad():
            t0 = time.time()
            pose_xy = self.fixed_point
            global_points_2onehot = torch.from_numpy(global_points_2onehot).float().to('cuda')
            t1 = time.time()
            # print(f"t0:{t1-t0}")
            for pose in push_actions:
        
                global_points_2onehot_ee = utils.Transform_Push2Fixed_point_onehot(global_points_2onehot, self.fixed_point, pose)
                # pose_points, _ = utils.push_gripper_pcd(pose[2], self.gripper_count) 
                pose_points = torch.cat([pose_xy, pose[2].unsqueeze(0)], dim=-1)
                pose_points = pose_points.unsqueeze(0)
                # pose_points_2onehot = torch.cat([pose_points, torch.tensor([0, 1], dtype=pose_points.dtype, device=pose_points.device).repeat(pose_points.size(0), 1)],dim=1)
                pose_points_3onehot = torch.cat([pose_points, torch.tensor([0, 0, 1], dtype=pose_points.dtype, device=pose_points.device).repeat(pose_points.size(0), 1)],dim=1)
 
                sence_points_2onehot = utils.furthest_point_sampling_onehot_p3d(global_points_2onehot_ee, n_samples=self.global_sample_count)
                sence_points_3onehot = torch.cat([sence_points_2onehot, torch.zeros((sence_points_2onehot.shape[0], 1), dtype=sence_points_2onehot.dtype, device=sence_points_2onehot.device)],dim = 1)
                fuse_points_3onehot = torch.cat([sence_points_3onehot, pose_points_3onehot],dim = 0) 
                select_idx = fuse_points_3onehot.shape[0] - 1
                fuse_points_3onehot = fuse_points_3onehot.T.to(dtype=torch.float32)
                uniform_fuse_points.append(fuse_points_3onehot)

            batch_g = torch.stack(uniform_fuse_points, dim=0)      
            # batch_g = batch_g.permute(0, 2, 1)   
            select_idx = torch.tensor([select_idx], dtype=int, device='cuda')           
            select_idxs = select_idx.repeat(batch_g.shape[0],1)
            batch_g = batch_g.to(device='cuda', dtype=torch.float32)
            preds = self.model(batch_g)
            probs = preds.squeeze(-1)                                    # [B,N]
            seed_prob = probs.gather(1, select_idxs.view(-1,1)).squeeze(1)

        # rank the actions score and select the best action to execute
        best_pre, best_action_idx = torch.max(seed_prob[:], dim=0)
        print(f"\033[37m best_push_pre = {best_pre}")

        best_push = push_actions[int(best_action_idx)]

        #---------visualize grasp state-----------------
        # grasp_pose = np.hstack([best_grasp.translation, R.from_matrix(best_grasp.rotation_matrix).as_quat()]) 
        # grasp_pose = torch.from_numpy(grasp_pose).float()
        # global_points_ee = utils.TransformPCD2EndLink(global_points, grasp_pose)
        # _, gripper_pcd = utils.grasp_pcd()
        # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
        # global_pcd = o3d.geometry.PointCloud()
        # global_pcd.points = o3d.utility.Vector3dVector(global_points_ee.cpu().numpy())
        # o3d.visualization.draw_geometries([frame, global_pcd, gripper_pcd])
        return best_push, best_action_idx
