from env.my_environment import Environment      
import argparse
import numpy as np
import random
import datetime
import torch
import utils
from grasp_evaluate import GraspEval
from env.constants import WORKSPACE_LIMITS
from tqdm import tqdm
import time
import open3d as o3d
env = Environment(gui=False)
my_seed = 1234
env.seed(my_seed)  
np.random.seed(my_seed)
torch.manual_seed(my_seed)
random.seed(my_seed)
num_episode = 5000
episode = 0
obj_num = 12
dir = 'your_grasp_model.pt'
graspnet = GraspEval(model_dir=dir, seed=my_seed, device='cuda') 
with tqdm(total=num_episode) as pbar:
    while True:
        #----environment reset----
        env.reset()
        if episode <= 5000:
            object_lis = env.add_objects(obj_num, WORKSPACE_LIMITS)
            target_obj = object_lis[0]
            object_pcds = []
            valid_obj_num = utils.is_in_workplace(env,target_obj) 
            if not valid_obj_num:
                continue
            ply_global_labels, seg= utils.get_global_label_pc(env, target_obj)
            ply_global = ply_global_labels[:, :3]

            pcd_mask = utils.get_fuse_pointcloud(env, target_obj)
            pcd_obj = utils.get_fuse_pointcloud(env, target_obj,id=1)

        grasp_evaluation, best_grasp_action, best_pre = graspnet.evalueate_grasp_without_interaction(ply_global, pcd_mask, push_flag=False)
        if grasp_evaluation:
            # success, grasped_obj_id, done_grasp = env.step(best_grasp_action)
            continue
        print('\033[31m Can not grasp at current state! \033[0m')
        print('------------------------------------------------------------')
        all_pcd = utils.get_all_obj_pointcloud(env, object_lis)
        target_pcd = all_pcd[0]

        i = 0
        for obj_pcd in all_pcd:
            # obj_pcd = utils.get_fuse_pointcloud(env, object_lis[i],id=1)
            valid_obj_num = utils.is_in_workplace(env, object_lis[i])
            object_near_target = utils.any_point_in_expanded_obb(target_pcd, obj_pcd)
            i += 1
            if valid_obj_num:
                if object_near_target:
                    object_pcds.append(obj_pcd)
        push_actions = utils.sample_push_action(ply_global, object_pcds)  # push_action is a 4x4 matrix Nx7
        push_actions_record = utils.transform_matrix2quat(push_actions)
        push_step = 1
        push_num = 10
        if len(push_actions) < push_num:
            push_num = len(push_actions)
        
        push_selected_id = np.random.choice(len(push_actions_record), push_num, replace=False)

        state_id = env.save_state()

        for i in push_selected_id:
            env.load_state(state_id)

            #----execute push action in a random way----
            push_action = push_actions[i]
            push_action_record = push_actions_record[i]
            push_is_valid_action, obj_is_move = env.push(push_action, target_obj)
            episode += 1
            valid_obj_num = utils.is_in_workplace(env,target_obj) 
            if not valid_obj_num:
                label = -1.0
            #----get next state to evaluate grasp prob----
            if valid_obj_num:
                next_pcd_obj = utils.get_fuse_pointcloud(env, target_obj,id=1)
                obj_points_change = abs(len(next_pcd_obj.points) - len(pcd_obj.points))
                # determine whether the push action makes the goal object around space change or not.
                if obj_is_move or obj_points_change > 2:
                    next_ply_global, _ = utils.get_global_pc(env)
                    print(f"\033[33m push make goal object move at episode:{episode} \033[0m")
                    next_pcd_mask = utils.get_fuse_pointcloud(env, target_obj)

                else:
                    print(f"\033[32m push don't make grasp state change at episode:{episode} \033[0m")
                    label = 0.0
                    #----record push action----
                    with open("env_data_collection/push_data/push_actions.txt", "a") as file:
                        file.write(
                        f"{push_action_record[0]} {push_action_record[1]} {push_action_record[2]} {push_action_record[3]} {push_action_record[4]} {push_action_record[5]} {push_action_record[6]}"
                        + "\n"
                        )
                    # ----record label----
                    with open('env_data_collection/push_data/labels.txt','a') as f:
                        f.write(f"{label}\n")
                    # ----record pc----
                    ply_global_name = f"env_data_collection/push_data/ply_global_labels/npy_global_{episode:05d}.npy"
                    np.save(ply_global_name,ply_global_labels)
                    pbar.update(1)
                    continue
                next_grasp_evaluation, next_best_grasp_action, next_best_pre = graspnet.evalueate_grasp_without_interaction(next_ply_global, next_pcd_mask, push_flag=False) 
                if next_grasp_evaluation:
                    label = torch.clip(torch.tensor(next_best_pre - 0.0),-1,1)
                else:
                    label = 0.0
            #----record push action---- 
            with open("env_data_collection/push_data/push_actions.txt", "a") as file:
                file.write(
                f"{push_action_record[0]} {push_action_record[1]} {push_action_record[2]} {push_action_record[3]} {push_action_record[4]} {push_action_record[5]} {push_action_record[6]}"
                + "\n"
                )
            #----record label----
            with open('env_data_collection/push_data/labels.txt','a') as f:
                f.write(f"{label}\n")
            #----record pc----
            ply_global_name = f"env_data_collection/push_data/ply_global_labels/npy_global_{episode:05d}.npy"
            # ply_obj_name = f"env_data_collection/push_data_v1/obj_npy/npy_obj_{episode:05d}.npy"

            np.save(ply_global_name,ply_global_labels)
            # np.save(ply_obj_name,ply_obj)

            pbar.update(1)
        if episode >= num_episode:
            break

