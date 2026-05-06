import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# from torch.utils.data import Dataset, DataLoader, random_split
from generate_grasp import Graspnet
from models.grasp_networks import Space_GraspFusion
import utils  
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import random
# ---------- Dataset ----------
class GraspEval():
    def __init__(self, model_dir, seed, device):
        # ---------- Arg Parser ----------
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        self.device = device
        self.model_dir = model_dir
        self.model = Space_GraspFusion(device='cuda').to(device)
        self.checkpoint = torch.load(self.model_dir, map_location='cuda')  # self.model_dir 为路径
        self.model.load_state_dict(self.checkpoint['model'])
        self.graspnet = Graspnet()

    def evalueate_grasp_actions(self, global_pc, obj_pc_mask, push_flag=True):
            T = None
            best_grasp = None
            best_pre = None
            uniform_fuse_points = []
            state_evaluate = False
            pose_points, _ = utils.grasp_pcd_bluenoise_like(n_target=170, oversample=2000, seed=55926)
            # pose_points, _ = utils.grasp_pcd_bluenoise_like(n_target=200, oversample=2000, seed=55926)
            pose_points = pose_points.to('cuda')
            self.model.eval()
            gg = []
            lengths = []
            with torch.no_grad():
                for i in range(2):
                    grasp_actions = self.graspnet.grasp_detection(obj_pc_mask)
                    lengths.append(len(grasp_actions))
                    gg.append(grasp_actions)

                # make grasp pose z axis oriented down
                if len(grasp_actions) < 1:
                    print('\033[32m No grasp pose be generated at current state! \033[0m')
                    print(f'\033[32m obj_pc_mask points num = {len(obj_pc_mask.points)} \033[0m')
                    return False, None, 0.0
                print(f"\033[32m grasp length = {len(gg[0])+ len(gg[1])} \033[0m")

                for i in range(2):
                    for j in range (len(gg[i])):
                        rotation = gg[i].rotation_matrices[j]
                        _, adjust_rotation = utils.adjust_pose_z_axis_to_down(rotation)
                        gg[i].rotation_matrices[j] = adjust_rotation

                global_points = torch.from_numpy(global_pc).float() 

                for pose in gg:
                    for grasp in pose :
                        pose_translation = grasp.translation
                        pose_translation[2] -= 0.01
                        pose_rotation = grasp.rotation_matrix
                        quat = R.from_matrix(pose_rotation).as_quat()
                        pose = np.hstack([pose_translation, quat])
                        pose = torch.from_numpy(pose).float()
                        global_points_ee = utils.TransformPCD2EndLink(global_points, pose)
                        sence_points = utils.fuse_state_torch_v2(global_points_ee, pose_points)
                        # sometime graspnet sample maybe not in goal-obj, so wo need to kick out that pose                    
                        if len(sence_points) == 0:
                            continue
                        sample_sence_points = utils.furthest_point_sampling_nocuda(sence_points,n_samples = 175)
                        sample_sence_points = sample_sence_points.to('cuda')
                        fuse_points = torch.cat([sample_sence_points,pose_points],dim = 0)
                        normalize_fuse_points,_,_ = utils.pc_normalize_grasp(fuse_points)
                        diff_labels = torch.cat([torch.ones((sample_sence_points.shape[0], 1),dtype=sample_sence_points.dtype, device=sample_sence_points.device), 
                                                torch.zeros((pose_points.shape[0], 1), dtype=pose_points.dtype, device=pose_points.device)],dim=0)
                        fuse_points_labels = torch.cat([normalize_fuse_points, diff_labels],dim = 1) 
                        uniform_fuse_point = fuse_points_labels.T.to(dtype=torch.float32)
                        uniform_fuse_points.append(uniform_fuse_point)

                batch_uniform_fuse_points = torch.stack(uniform_fuse_points, dim=0) # Nx3 -> 10xNx3
                # batch_uniform_fuse_points = batch_uniform_fuse_points.transpose(1, 2) # 10xNx3 -> 10x3xN
                pred = self.model(batch_uniform_fuse_points)
                pred = F.softmax(pred, dim=1)
                pred_class = pred.data.max(1, keepdim=True)[1]

            action_idxs = (pred_class == 1).nonzero(as_tuple=True)[0]
            if len(action_idxs) > 0:
                # state_evaluate = True
            # rank the succeessful actions and select the best action to execute
                best_pre, best_action_idx = torch.max(pred[:,1], dim=0)
                if best_pre >= 0.75:
                    state_evaluate = True
                if push_flag:
                    state_evaluate = True
                
                print(f"\033[36m best_grasp_pre = {best_pre} \033[0m")
                if best_action_idx + 1 <= lengths[0]:
                    best_grasp = gg[0][int(best_action_idx)]
                else:
                    best_action_idx = best_action_idx - lengths[0]
                    best_grasp = gg[1][int(best_action_idx)]
                T = np.eye(4)
                T[:3, :3] = best_grasp.rotation_matrix
                T[:3, 3] = best_grasp.translation
                T[2, 3] -= 0.01
                #----------------visualize grasp state-----------------
                # grasp_pose = np.hstack([best_grasp.translation, R.from_matrix(best_grasp.rotation_matrix).as_quat()]) 
                # grasp_pose = torch.from_numpy(grasp_pose).float()
                # global_points_ee = utils.TransformPCD2EndLink(global_points, grasp_pose)
                # _, gripper_pcd = utils.grasp_pcd()
                # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
                # global_pcd = o3d.geometry.PointCloud()
                # global_pcd.points = o3d.utility.Vector3dVector(global_points_ee.cpu().numpy())
                # o3d.visualization.draw_geometries([frame, global_pcd, gripper_pcd])
                
                return state_evaluate, T, best_pre
            else:
                print('\033[31m No valid grasp pose at that state! \033[0m')
                return False, None, 0.0

    def evalueate_grasp_without_interaction(self, global_points, obj_pc_mask, push_flag=True):
        # preds = []
        # pred_classification = [] 
        T = None
        best_grasp = None
        best_pre = None
        uniform_fuse_points = []
        state_evaluate = False
        pose_points, _ = utils.grasp_pcd_bluenoise_like(n_target=170, oversample=2000, seed=55926)
        pose_points = pose_points.to('cuda')
        self.model.eval()
        gg = []
        lengths = []
        with torch.no_grad():
            for i in range(2):
                grasp_actions = self.graspnet.grasp_detection(obj_pc_mask)
                lengths.append(len(grasp_actions))
                gg.append(grasp_actions)

            # make grasp pose z axis oriented down
            if sum(lengths) < 1:
                print('\033[32m No grasp pose be generated at current state! \033[0m')
                print(f'\033[32m obj_pc_mask points num = {len(obj_pc_mask.points)} \033[0m')
               
            print(f"\033[32m grasp length = {len(grasp_actions)} \033[0m")

            for i in range(2):
                for j in range (len(gg[i])):
                    rotation = gg[i].rotation_matrices[j]
                    _, adjust_rotation = utils.adjust_pose_z_axis_to_down(rotation)
                    gg[i].rotation_matrices[j] = adjust_rotation

            global_points = torch.from_numpy(global_points).float() 

            for pose in gg:
                for grasp in pose :
                    pose_translation = grasp.translation
                    pose_rotation = grasp.rotation_matrix
                    quat = R.from_matrix(pose_rotation).as_quat()
                    pose = np.hstack([pose_translation,quat])
                    pose = torch.from_numpy(pose).float()
                    global_points_ee = utils.TransformPCD2EndLink(global_points, pose)
                    sence_points = utils.fuse_state_torch_v2(global_points_ee, pose_points)
                    # sometime graspnet sample maybe not in goal-obj, so wo need to kick out that pose                    
                    if len(sence_points) == 0:
                        continue
                    sample_sence_points = utils.furthest_point_sampling_nocuda(sence_points,n_samples = 345)
                    sample_sence_points = sample_sence_points.to('cuda')
                    normalize_fuse_points,_,_ = utils.pc_normalize_grasp(sample_sence_points)
                    normalize_fuse_points = normalize_fuse_points.T.to(dtype=torch.float32)  
                    uniform_fuse_points.append(normalize_fuse_points)

            if len(uniform_fuse_points) == 0:
                return False, None, None
            batch_uniform_fuse_points = torch.stack(uniform_fuse_points, dim=0) # Nx3 -> 10xNx3
            # batch_uniform_fuse_points = batch_uniform_fuse_points.transpose(1, 2) # 10xNx3 -> 10x3xN
            pred = self.model(batch_uniform_fuse_points)
            pred = F.softmax(pred, dim=1)
            pred_class = pred.data.max(1, keepdim=True)[1]

        action_idxs = (pred_class == 1).nonzero(as_tuple=True)[0]
        if len(action_idxs) > 0:
        # rank the succeessful actions and select the best action to execute
            best_pre, best_action_idx = torch.max(pred[:,1], dim=0)
            if best_pre >= 0.75:
                state_evaluate = True
            if push_flag:
                state_evaluate = True
            
            print(f"\033[36m best_grasp_pre = {best_pre} \033[0m")
            if best_action_idx + 1 <= lengths[0]:
                best_grasp = gg[0][int(best_action_idx)]
            else:
                best_action_idx = best_action_idx - lengths[0]
                best_grasp = gg[1][int(best_action_idx)]
            T = np.eye(4)
            T[:3, :3] = best_grasp.rotation_matrix
            T[:3, 3] = best_grasp.translation

            #----------------visualize grasp state-----------------
            # grasp_pose = np.hstack([best_grasp.translation, R.from_matrix(best_grasp.rotation_matrix).as_quat()]) 
            # grasp_pose = torch.from_numpy(grasp_pose).float()
            # global_points_ee = utils.TransformPCD2EndLink(global_points, grasp_pose)
            # _, gripper_pcd = utils.grasp_pcd()
            # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
            # global_pcd = o3d.geometry.PointCloud()
            # global_pcd.points = o3d.utility.Vector3dVector(global_points_ee.cpu().numpy())
            # o3d.visualization.draw_geometries([frame, global_pcd, gripper_pcd])
            return state_evaluate, T, best_pre
        else:
            print('\033[31m No valid grasp pose at that state! \033[0m')
            return False, None, 0.0
                
                
                    
           
                

