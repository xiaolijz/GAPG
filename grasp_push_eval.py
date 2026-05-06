import argparse
import numpy as np
import random
import datetime
import torch
import utils
from env.constants import WORKSPACE_LIMITS
from grasp_evaluate import GraspEval
from push_evaluate import PushEval
import time
import open3d as o3d
from env.my_environment import Environment   

my_seed = 1234 # set random seed for test
env = Environment(gui=True)
env.seed(my_seed)
np.random.seed(my_seed)
torch.manual_seed(my_seed)
random.seed(my_seed)

obj_num = 20
epoch = 0
episode_index_record = []
#---------we use three index to evalue our system---------------
grasp_pt_dir = 'pretrain_models/grasp_model.pt'
graspnet = GraspEval(model_dir=grasp_pt_dir, seed=my_seed, device='cuda') 
push_pt_dir = 'pretrain_models/push_model.pt'
pushnet = PushEval(model_dir=push_pt_dir, seed=my_seed, device='cuda')
#--------challenge test dir
challenge_txt_dir = 'assets/challenge/09.txt'
while epoch < 30:
    #----environment reset----
    env.reset()
    flag = 1
    episode = 0
    complete_rate = 0
    successful_grasp_rate = 0
    average_move = 0
    # test in random scene
    object_lis = env.add_objects(obj_num, WORKSPACE_LIMITS)
    # test in challenge scene
    # object_lis = env.load_objects_from_txt(challenge_txt_dir)
    target_obj = object_lis[0]
    valid_obj_num = utils.is_in_workplace(env,target_obj) # ensure goal obj in workspace
    if not valid_obj_num:
        continue
    ply_global, seg = utils.get_global_pc(env)
    pcd_mask = utils.get_fuse_pointcloud(env, target_obj)
    
    target_pcd = utils.get_fuse_pointcloud(env, target_obj, id=1)
    if len(target_pcd.points) <= 0:
        print('unseen target')
        continue
    epoch += 1
    print(f'\033[37m current epoch is {epoch} \033[0m')
    while episode <= 10:
        done = True
        object_pcds = []
        while done:
            grasp_evaluation, best_grasp_action, best_pre = graspnet.evalueate_grasp_actions(ply_global, pcd_mask, push_flag=False)
            # successful_grasp_rate += 1
            if grasp_evaluation:
                success, grasped_obj_id, done_grasp = env.step(best_grasp_action)
                successful_grasp_rate += 1
                average_move += 1
                if grasped_obj_id is not None :
                    if target_obj in grasped_obj_id:
                        complete_rate += 1
                        print('-------------------------Successful Grasp!!!---------------------------------')
                        break
                else:
                    print('-------------------------Unsuccessful Grasp!!!-------------------------------')
                    ply_global, seg= utils.get_global_pc(env)
                    pcd_mask = utils.get_fuse_pointcloud(env, target_obj)
                    target_pcd = utils.get_fuse_pointcloud(env, object_lis[0], id=1)
            else:
                done = False
        if done:
            break
        if not done:
            all_pcd = utils.get_all_obj_pointcloud(env, object_lis)
            i = 0
            for obj_pcd in all_pcd:
                # obj_pcd = utils.get_fuse_pointcloud(env, object_lis[i],id=1)
                valid_obj_num = utils.is_in_workplace(env, object_lis[i])
                object_near_target = utils.any_point_in_expanded_obb(target_pcd, obj_pcd)
                i += 1
                if valid_obj_num:
                    if object_near_target:
                        object_pcds.append(obj_pcd)
            #--------------------we sample push actions from global pc-------------------
            ply_global_labels, seg = utils.get_global_label_pc(env, target_obj)
            push_actions = utils.sample_push_action(ply_global, object_pcds)
            push_actions_quat = utils.transform_matrix2quat(push_actions)
            _, idx = pushnet.evalueate_push_actions(ply_global_labels, push_actions_quat)
            best_push_action = push_actions[idx]
            _, _ = env.push(best_push_action, target_obj)
            valid_obj_num = utils.is_in_workplace(env, target_obj) # ensure goal obj in workspace
            if not valid_obj_num:
                flag = 0
                epoch -= 1
                break
            average_move += 1
            episode += 1
            #-----------graspnet evaluate the next state------------------------
            ply_global, seg= utils.get_global_pc(env)
            pcd_mask = utils.get_fuse_pointcloud(env, target_obj)
            target_pcd = utils.get_fuse_pointcloud(env, object_lis[0],id=1)

    if flag == 1:
        episode_index_record.append((successful_grasp_rate, complete_rate, average_move))
# successful_grasp_rate = float(successful_grasp_rate) / complete_rate
successful_grasp_rates, complete_rates, average_moves = zip(*episode_index_record)
successful_grasp_rate = sum(successful_grasp_rates)
complete_rate = sum(complete_rates)
average_move= sum(average_moves)
print(f"complete_rate:{complete_rate},successful_grasp_rate:{successful_grasp_rate}")

successful_grasp_rate = float(complete_rate) / successful_grasp_rate
complete_rate = float(complete_rate) / 30.0
average_move = float(average_move) / 30.0
print(f'\033[34m successful_grasp_rate = {successful_grasp_rate},\n complete_rate = {complete_rate}, \n average_move = {average_move} \033[0m')
