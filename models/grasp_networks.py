from collections import OrderedDict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_encoder import Encoder_PointCloud,Encoder_PointCloud_SAC,Encoder_PointCloud_Type
import numpy as np

class Space_GraspFusion(nn.Module):
    def __init__(self, device,k=2):
        super().__init__()
        
        self.device = device

        self.pointnet = Encoder_PointCloud_Type() 

        self.fc1 = nn.Linear(1024, 512) 
        self.drop1 = nn.Dropout(0.4) 
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, k)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(128)

        self.relu = nn.ReLU()

    def encode_SpacePointCloud(self, x):
        spacepointcloud = self.pointnet(x.to(self.device))
        return spacepointcloud
    

    def forward(self, pointcloud):

        space_feat = self.encode_SpacePointCloud(pointcloud)
        space_feat = space_feat.squeeze(-1)
        # B,N,C = space_feat.shape
        x = F.relu(self.bn1(self.fc1(space_feat)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        # return F.log_softmax(x, dim=-1)
        return x
    


