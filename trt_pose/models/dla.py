import sys
import os
import torch
from .common import *
import timm
# from torch.autograd import Variable
# import torch.onnx

# sys.path.append(os.path.join(os.path.dirname(__file__), 'dla'))

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class DlaBackbone(torch.nn.Module):
    def __init__(self, dla):
        super(DlaBackbone, self).__init__()
        self.dla = dla
        
    def forward(self, x):
        x = self.dla.base_layer(x)
        x = self.dla.level0(x)
        x = self.dla.level1(x)
        x = self.dla.level2(x)
        x = self.dla.level3(x)

        return x

def _dla_pose_att(cmap_channels, paf_channels, upsample_channels, dla, feature_channels, num_upsample, num_flat):
    model = torch.nn.Sequential(
        DlaBackbone(dla),
        CmapPafHeadAttention(feature_channels, cmap_channels, paf_channels, upsample_channels, num_upsample=num_upsample, num_flat=num_flat)
    )
    return model

def dla34up_pose(cmap_channels, paf_channels, upsample_channels=512, pretrained=True, num_upsample=2, num_flat=0):
    dla = timm.create_model('dla34', pretrained=pretrained)
    dla.level4 = Identity()
    dla.level5 = Identity()
    dla.global_pool = Identity()
    dla.fc = Identity()
    return _dla_pose_att(cmap_channels, paf_channels, upsample_channels, dla, 128, num_upsample, num_flat)

# model = dla34up_pose(18,42)
# print(model)
# dummy_input = Variable(torch.randn(1, 3, 256, 256))
# torch.onnx.export(model, dummy_input, "model_dla34.onnx")
