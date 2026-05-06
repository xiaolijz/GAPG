import os
from env.my_environment import Environment      
import argparse
import numpy as np
import random
import datetime
import torch
import utils
from generate_grasp import Graspnet
from env.constants import WORKSPACE_LIMITS
from tqdm import tqdm
import time
env = Environment(gui=False)
env.seed(1111)
np.random.seed(1111)
torch.manual_seed(1111)
num_episode = 15000
episode = 0
graspnet = Graspnet() 
save_dir = "env_data_collection/grasp_data/ply_global"
os.makedirs(save_dir, exist_ok=True)
with tqdm(total=num_episode) as pbar:
    while True:
        #----reset environment----
        env.reset()
        object_lis = env.add_objects(15, WORKSPACE_LIMITS)
        random_numbers = np.random.choice(15, size=5, replace=False)
        num_obj_grasp = 5
        for i in range(num_obj_grasp):
            #----get a random target----
            obj_num = object_lis[random_numbers[i]]
            valid_obj_num = utils.is_in_workplace(env,obj_num) # Determine if the target object is in the workspace
            if not valid_obj_num:
                continue
            #----get global point cloud----
            ply_global, _= utils.get_global_pc(env)
            #----get grasp poses----
            pcd_mask = utils.get_fuse_pointcloud(env, obj_num)
            poses = graspnet.grasp_detection(pcd_mask)
            if poses is None or len(poses.translations) == 0:
                continue
            #----record pc----
            ply_global_name = os.path.join(save_dir, f"global_{episode:05d}.npy")
            np.save(ply_global_name, ply_global)

            pose_num = np.random.randint(len(poses))
            pose_translation = poses.translations[pose_num]
            pose_translation[2] = pose_translation[2] - 0.005
            pose_rotation = poses.rotation_matrices[pose_num]
            quaternion, pose_rotation = utils.adjust_pose_z_axis_to_down(pose_rotation)
            pose = np.eye(4)
            pose[:3,:3] = pose_rotation
            pose[:3,3] = pose_translation
            record_pose = np.hstack([pose_translation,quaternion])
            #----record grasp poses----
            with open("env_data_collection/grasp_data/poses.txt", "a") as file:
                file.write(
                    f"{record_pose[0]} {record_pose[1]} {record_pose[2]} {record_pose[3]} {record_pose[4]} {record_pose[5]} {record_pose[6]}"
                    + "\n"
                )
            #----execute grasp----
            success, grasped_obj_id, done = env.step(pose)
            #----record label----
            with open('env_data_collection/grasp_data/labels.txt','a') as f:
                f.write(f"{int(obj_num==grasped_obj_id)}\n")
            
            # np.save(f"env_data_collection/ground_true_dateset/mask/mask_{episode:05d}.npy", mask)
            episode += 1
            pbar.update(1)
        if episode > num_episode:
            break

