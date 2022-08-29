import torch
from torch import nn


# Point-wise attention for each voxel
class PALayer(nn.Module):
    def __init__(self, dim_pa, reduction_pa):
        super(PALayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim_pa, dim_pa // reduction_pa),
            nn.ReLU(inplace=True),
            nn.Linear(dim_pa // reduction_pa, dim_pa)
        )

    def forward(self, x):
        b, w, _ = x.size()
        y = torch.max(x, dim=2, keepdim=True)[0].view(b, w)
        out1 = self.fc(y).view(b, w, 1)
        return out1


# Channel-wise attention for each voxel
class CALayer(nn.Module):
    def __init__(self, dim_ca, reduction_ca):
        super(CALayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim_ca, dim_ca // reduction_ca),
            nn.ReLU(inplace=True),
            nn.Linear(dim_ca // reduction_ca, dim_ca)
        )

    def forward(self, x):
        b, _, c = x.size()
        y = torch.max(x, dim=1, keepdim=True)[0].view(b, c)
        y = self.fc(y).view(b, 1, c)
        return y


# Point-wise attention for each voxel
class PACALayer(nn.Module):
    def __init__(self, dim_ca, dim_pa, reduction_r):
        super(PACALayer, self).__init__()
        self.pa = PALayer(dim_pa, dim_pa // reduction_r)
        self.ca = CALayer(dim_ca, dim_ca // reduction_r)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        pa_weight = self.pa(x)
        ca_weight = self.ca(x)
        paca_weight = torch.mul(pa_weight, ca_weight)
        paca_normal_weight = self.sig(paca_weight)
        out = torch.mul(x, paca_normal_weight)
        return out, paca_normal_weight


# Voxel-wise attention for each voxel
class VALayer(nn.Module):
    def __init__(self, c_num, p_num):
        super(VALayer, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(c_num + 3, 1),
            nn.ReLU(inplace=True)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(p_num, 1),
            nn.ReLU(inplace=True)
        )

        self.sigmod = nn.Sigmoid()

    def forward(self, voxel_center, paca_feat):
        '''
        :param voxel_center: size (K,1,3)
        :param SACA_Feat: size (K,N,C)
        :return: voxel_attention_weight: size (K,1,1)
        '''
        voxel_center_repeat = voxel_center.repeat(1, paca_feat.shape[1], 1)
        # print(voxel_center_repeat.shape)
        voxel_feat_concat = torch.cat([paca_feat, voxel_center_repeat], dim=-1)  # K,N,C---> K,N,(C+3)

        feat_2 = self.fc1(voxel_feat_concat)  # K,N,(C+3)--->K,N,1
        feat_2 = feat_2.permute(0, 2, 1).contiguous()  # K,N,1--->K,1,N

        voxel_feat_concat = self.fc2(feat_2)  # K,1,N--->K,1,1

        voxel_attention_weight = self.sigmod(voxel_feat_concat)  # K,1,1

        return voxel_attention_weight


class VoxelFeature_TA(nn.Module):
    def __init__(self, dim_ca=9, dim_pa=100,
                 reduction_r=8, boost_c_dim=64,
                 use_paca_weight=False):
        super(VoxelFeature_TA, self).__init__()
        self.PACALayer1 = PACALayer(dim_ca=dim_ca, dim_pa=dim_pa, reduction_r=reduction_r)
        self.PACALayer2 = PACALayer(dim_ca=boost_c_dim, dim_pa=dim_pa, reduction_r=reduction_r)
        self.voxel_attention1 = VALayer(c_num=dim_ca, p_num=dim_pa)
        self.voxel_attention2 = VALayer(c_num=boost_c_dim, p_num=dim_pa)
        self.use_paca_weight = use_paca_weight
        self.FC1 = nn.Sequential(
            nn.Linear(2 * dim_ca, boost_c_dim),
            nn.ReLU(inplace=True),
        )
        self.FC2 = nn.Sequential(
            nn.Linear(boost_c_dim, boost_c_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, voxel_center, x):
        paca1, paca_normal_weight1 = self.PACALayer1(x)
        voxel_attention1 = self.voxel_attention1(voxel_center, paca1)
        if self.use_paca_weight:
            paca1_feat = voxel_attention1 * paca1 * paca_normal_weight1
        else:
            paca1_feat = voxel_attention1 * paca1
        out1 = torch.cat([paca1_feat, x], dim=2)
        out1 = self.FC1(out1)

        paca2, paca_normal_weight2 = self.PACALayer2(out1)
        voxel_attention2 = self.voxel_attention2(voxel_center, paca2)
        if self.use_paca_weight:
            paca2_feat = voxel_attention2 * paca2 * paca_normal_weight2
        else:
            paca2_feat = voxel_attention2 * paca2
        out2 = out1 + paca2_feat
        out = self.FC2(out2)

        return out

