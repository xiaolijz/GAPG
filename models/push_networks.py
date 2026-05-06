from collections import OrderedDict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_encoder import Encoder_PointCloud,Encoder_PointCloud_Type_pose_obj,Encoder_PointCloud_Type_pose_sence, Encoder_PointCloud_Type
import numpy as np
from models.pointnet2_utils import (
    PointNetSetAbstraction,
    PointNetFeaturePropagation,
)
class Space_Push_Fusion(nn.Module):
    def __init__(self, device,k=1):
        super().__init__()
        
        self.device = device

        self.pointnet_global = Encoder_PointCloud_Type() 
        self.pointnet_obj = Encoder_PointCloud()
        self.fc1 = nn.Linear(1024, 512) 
        self.drop1 = nn.Dropout(0.4) 
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, k) # In this stage, push model as a score evaluation tool
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(128)

        self.relu = nn.ReLU()

    def encode_SpacePointCloud(self, x):
        spacepointcloud = self.pointnet_global(x.to(self.device))
        return spacepointcloud 
    
    def encode_goal_obj_pc(self, x):
        return self.pointnet_obj(x.to(self.device))

    def forward(self, global_pc, obj_pc):

        global_feat = self.encode_SpacePointCloud(global_pc)
        global_feat = global_feat.squeeze(-1) # Bx512
        obj_feat = self.encode_goal_obj_pc(obj_pc)
        obj_feat = obj_feat.squeeze(-1) # Bx512
        # B,N,C = space_feat.shape
        space_feat = torch.cat([global_feat, obj_feat],dim=1) # Bx1024
        x = F.relu(self.bn1(self.fc1(space_feat)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        # return F.log_softmax(x, dim=-1)
        return x
    

class Space_Push_Fusion_v2(nn.Module):
    def __init__(self, device,k=1):
        super().__init__()
        
        self.device = device

        self.pointnet_global = Encoder_PointCloud_Type_pose_sence(width=512,additional_channel=3) # 512
        self.pointnet_obj = Encoder_PointCloud_Type_pose_obj(width=512,additional_channel=2) # 512
        self.fc1 = nn.Linear(1024, 512) 
        self.drop1 = nn.Dropout(0.4) 
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, k) # In this stage, push model as a score evaluation tool
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(128)

        self.relu = nn.ReLU()

    def encode_SpacePointCloud(self, x):
        spacepointcloud = self.pointnet_global(x.to(self.device))
        return spacepointcloud 
    
    def encode_goal_obj_pc(self, x):
        return self.pointnet_obj(x.to(self.device))

    def forward(self, global_pc, obj_pc):

        global_feat = self.encode_SpacePointCloud(global_pc)
        global_feat = global_feat.squeeze(-1) # Bx512
        obj_feat = self.encode_goal_obj_pc(obj_pc)
        obj_feat = obj_feat.squeeze(-1) # Bx512
        # B,N,C = space_feat.shape
        space_feat = torch.cat([global_feat, obj_feat],dim=1) # Bx1024
        x = F.relu(self.bn1(self.fc1(space_feat)))
        x = self.drop1(x) 
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        # return F.log_softmax(x, dim=-1)
        return x
    
class Push_model(nn.Module):
    def __init__(self, width=1024,  additional_channel = 0):
        super(Push_model, self).__init__()

        self.sa1 = PointNetSetAbstraction(
            npoint=512,
            radius=0.2,
            nsample=32,
            in_channel=6 + additional_channel,
            mlp=[64, 64, 128],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128,
            radius=0.4,
            nsample=64,
            in_channel=128 + 3,
            mlp=[128, 128, 256],
            group_all=False,
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=256 + 3,
            mlp=[256, 512, width],
            group_all=True,
        )
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(
            in_channel=128 + 6 + additional_channel, mlp=[128, 128, 128]
        )
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, 1, 1)

    def forward(self, xyz):
        # Set Abstraction layers
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(
            l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points
        )
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        output = self.drop1(feat)
        output = self.conv2(output)
        output = torch.sigmoid(output)
        output = output.permute(0, 2, 1)
        return output

class Push_Model_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.crit = nn.BCELoss()     

    def forward(self, pred, seed_idx, target):
        # pred: [B,N,1] -> [B,N]
        prob = pred.squeeze(-1)
        seed_idx = seed_idx.view(-1,1)          # [B,1]
        seed_prob = prob.gather(1, seed_idx).squeeze(1)  # [B]
        target = target.float().view(-1)        # [B]
        return self.crit(seed_prob, target)
