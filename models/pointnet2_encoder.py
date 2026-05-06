import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pointnet2_utils import (
    PointNetSetAbstraction,
    PointNetFeaturePropagation,
)
class Encoder_PointCloud(nn.Module):
    def __init__(self, normal_channel=False):
        super(Encoder_PointCloud, self).__init__()
        self.normal_channel = normal_channel
        additional_channel = 3 if normal_channel else 0

        self.sa1 = PointNetSetAbstraction(
            npoint=128,
            radius=0.2,
            nsample=32,
            in_channel=6,
            mlp=[64, 64, 128],
            group_all=False,
        )

        self.sa2 = PointNetSetAbstraction(
            npoint=64,
            radius=0.4,
            nsample=64,
            in_channel=128 + 3,
            mlp=[128, 256, 256],
            group_all=False,
        )

        self.sa3 = PointNetSetAbstraction(
            npoint=None, 
            radius=None, 
            nsample=None,
            in_channel=256 + 3, 
            mlp=[256, 512, 1024], 
            group_all=True)

    def forward(self, xyz):

        if self.normal_channel:
            l0_xyz = xyz[:, :3, :]           
            l0_points = xyz                  
        else:
            l0_xyz = xyz[:, :3, :]          
            l0_points = l0_xyz            # [B, C, N]


        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        return l3_points 


class Encoder_PointCloud_Type(nn.Module):
    def __init__(self, add_channel=True):
        super(Encoder_PointCloud_Type, self).__init__()
        # self.normal_channel = normal_channel
        additional_channel = 1 if add_channel else 0

        self.sa1 = PointNetSetAbstraction(
            npoint=128,
            radius=0.2,
            nsample=32,
            in_channel=6 + additional_channel,
            mlp=[64, 64, 128],
            group_all=False,
        )

        self.sa2 = PointNetSetAbstraction(
            npoint=64,
            radius=0.4,
            nsample=64,
            in_channel=128 + 3,
            mlp=[128, 256, 256],
            group_all=False,
        )

        self.sa3 = PointNetSetAbstraction(
            npoint=None, 
            radius=None, 
            nsample=None,
            in_channel=256 + 3, 
            mlp=[256, 512, 1024], 
            group_all=True)

    def forward(self, xyz):
        # xyz-Bx4xN        
        l0_xyz = xyz[:, :3, :]  # [B, C, N]
        l0_points = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        return l3_points 
    

class Encoder_PointCloud_Type_pose_sence(nn.Module):
    def __init__(self, width=1024,  additional_channel = 0):
        super(Encoder_PointCloud_Type_pose_sence, self).__init__()
        # self.normal_channel = normal_channel

        self.sa1 = PointNetSetAbstraction(
            npoint=256,
            radius=0.25,
            nsample=48,
            in_channel=6 + additional_channel,
            mlp=[64, 64, 128],
            group_all=False,
        )

        self.sa2 = PointNetSetAbstraction(
            npoint=64,
            radius=0.5,
            nsample=64,
            in_channel=128 + 3,
            mlp=[128, 256, 256],
            group_all=False,
        )

        self.sa3 = PointNetSetAbstraction(
            npoint=None, 
            radius=None, 
            nsample=None,
            in_channel=256 + 3, 
            mlp=[256, 512, width], 
            group_all=True)

    def forward(self, xyz):
        # xyz-Bx4xN        
        l0_xyz = xyz[:, :3, :]  # [B, C, N]
        l0_points = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        return l3_points 

class Encoder_PointCloud_Type_pose_obj(nn.Module):
    def __init__(self, width=1024,  additional_channel = 0):
        super(Encoder_PointCloud_Type_pose_obj, self).__init__()
        # self.normal_channel = normal_channel

        self.sa1 = PointNetSetAbstraction(
            npoint=64,
            radius=0.3,
            nsample=32,
            in_channel=6 + additional_channel,
            mlp=[64, 64, 128],
            group_all=False,
        )

        self.sa2 = PointNetSetAbstraction(
            npoint=32,
            radius=0.6,
            nsample=48,
            in_channel=128 + 3,
            mlp=[128, 256, 256],
            group_all=False,
        )

        self.sa3 = PointNetSetAbstraction(
            npoint=None, 
            radius=None, 
            nsample=None,
            in_channel=256 + 3, 
            mlp=[256, 512, width], 
            group_all=True)

    def forward(self, xyz):
        # xyz-Bx4xN        
        l0_xyz = xyz[:, :3, :]  # [B, C, N]
        l0_points = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        return l3_points 

class Encoder_PointCloud2(nn.Module):
    def __init__(self, normal_channel=False):
        super(Encoder_PointCloud2, self).__init__()
        self.normal_channel = normal_channel
        additional_channel = 3 if normal_channel else 0

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
            mlp=[256, 512, 1024],
            group_all=True,
        )
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(
            in_channel=128 + 6 + additional_channel, mlp=[128, 256]
        )

    def forward(self, xyz):
        B, C, N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]
        else:
            l0_points = xyz
            l0_xyz = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(
            l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points
        )  # [B, 128, N]

        point_features = l0_points.permute(2, 0, 1)  # [B, N, 256]
        return point_features
    
class Encoder_PointCloud_SAC(nn.Module):
    def __init__(self, width, normal_channel=False):
        super(Encoder_PointCloud_SAC, self).__init__()
        self.normal_channel = normal_channel
        additional_channel = 3 if normal_channel else 0
        self.width = width
        self.sa1 = PointNetSetAbstraction(
            npoint=128,
            radius=0.2,
            nsample=32,
            in_channel=6,
            mlp=[64, 64, 128],
            group_all=False,
        )

        self.sa2 = PointNetSetAbstraction(
            npoint=64,
            radius=0.4,
            nsample=32,
            in_channel=128 + 3,
            mlp=[128, 256],
            group_all=False,
        )

        self.sa3 = PointNetSetAbstraction(
            npoint=None, 
            radius=None, 
            nsample=None,
            in_channel=256 + 3, 
            mlp=[256, self.width], 
            group_all=True)

    def forward(self, xyz):

        if self.normal_channel:
            l0_xyz = xyz[:, :3, :]           # [B, 3, N]
            l0_points = xyz                  # [B, 6, N]
        else:
            l0_xyz = xyz[:, :3, :]           # [B, 3, N]
            l0_points = l0_xyz            # [B, N, 3]


        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        return l3_points 
    
class Encoder_PointCloud_Push_Type(nn.Module):

    def __init__(self, additional_channel=3):
        super(Encoder_PointCloud_Push_Type, self).__init__()

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
            mlp=[256, 512, 1024],
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
        B, C, N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]
        else:
            l0_points = xyz
            l0_xyz = xyz
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

class Encoder_PointCloud_Push_Type_Loss(nn.Module):

    def __init__(self):

        super(Encoder_PointCloud_Push_Type_Loss, self).__init__()

    def forward(self, pred, target):
        """
        Args:
        - pred : (B, N, 1)  ->  prediction output of model.
        - target : (B, 1)   ->  target value.

        Loss Function :
        - binary_cross_entropy

        Output:
        - loss
        """
        # get shape parameter.
        B, N, _ = pred.shape
        # shape of 'pred_selected' is (B,), indicating the probability of chosen point.
        pred_selected = pred[torch.arange(B), 0, 0]
        # print("prediction_probability is ", pred_selected)
        # change the shape of target from (B, 1) to (B,)
        target = target.view(-1)
        # use binary cross entropy to calculate loss.
        loss = F.binary_cross_entropy(pred_selected, target)

        return loss