"""
PointPillars fork from SECOND.
Code written by Alex Lang and Oscar Beijbom, 2018.
Licensed under MIT License [see LICENSE].
"""

import torch
from torch import nn
from torch.nn import functional as F
from second.pytorch.models.TANet import VoxelFeature_TA
from second.pytorch.models.voxel_encoder import get_paddings_indicator, register_vfe
from second.pytorch.models.middle import register_middle
from torchplus.nn import Empty
from torchplus.tools import change_default_args
import numpy as np


class PCAttention(nn.Module):
    def __init__(self, gate_channels, reduction_rate, pool_types=['max', 'mean'], activation=nn.ReLU(),
                 channel_mean=False):
        super(PCAttention, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Linear(gate_channels, gate_channels // reduction_rate),
            activation,
            nn.Linear(gate_channels // reduction_rate, gate_channels)
        )
        self.pool_types = pool_types
        print('self.pool_types', self.pool_types)

        self.max_pool = nn.AdaptiveMaxPool2d((1, None))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, None)) #if not channel_mean else GetChannelMean(keepdim=True)
        print('self.max_pool, self.avg_pool',self.max_pool, self.avg_pool)
    def forward(self, x):
        '''
        # shape [n_voxels, channels, n_points] for point-wise attention
        # shape [n_voxels, n_points, channels] for channels-wise attention
        '''
        attention_sum = None
        for pool_type in self.pool_types:
            # [n_voxels, 1, n_points]
            if pool_type == 'max':
                max_pool = self.max_pool(x)
                attention_raw = self.mlp(max_pool)
            elif pool_type == 'mean':
                avg_pool = self.avg_pool(x)
                attention_raw = self.mlp(avg_pool)
            if attention_sum is None:
                attention_sum = attention_raw
            else:
                attention_sum += attention_raw
        scale = torch.sigmoid(attention_sum).permute(0, 2, 1)
        # scale = attention_sum.permute(0, 2, 1)
        return scale


class GetChannelMean(nn.Module):
    def __init__(self, keepdim=True):
        super(GetChannelMean, self).__init__()
        self.keepdim = keepdim

    def forward(self, x):  # [n_voxels, n_points, n_channels]
        x = x.permute(0, 2, 1)  # [n_voxels, n_channels, n_points]
        sum = x.sum(dim=-1)
        cnt = (x != 0).sum(-1).type(torch.float)
        cnt[cnt == 0] = 1  # replace 0 to 1 to avoid divide by 0
        mean = torch.true_divide(sum, cnt)  # [n_voxels, n_channels]
        if self.keepdim:
            return mean.unsqueeze(-1).permute(0, 2, 1)  # [n_voxels, n_channels, 1] -> [n_voxels, 1, n_channels]
        return mean


class TaskAware(nn.Module):
    def __init__(self, channels, reduction_rate=8, k=2, pool_types=['max','mean']):
        super(TaskAware, self).__init__()
        self.channels = channels
        self.k = k
        self.fc = nn.Sequential(nn.Linear(channels, channels // reduction_rate),
                                nn.ReLU(inplace=True),
                                nn.Linear(channels // reduction_rate, 2 * k * channels),
                                )
        self.sigmoid = nn.Sigmoid()
        self.register_buffer('lambdas', torch.Tensor([1.] * k + [0.5] * k).float())
        self.register_buffer('init_v', torch.Tensor([1.] + [0.] * (2 * k - 1)).float())
        # self.get_mean = GetChannelMean(keepdim=False)

    def get_relu_coefs(self, x):  # [n_voxels, n_points, n_channels]
        mean = torch.mean(x, dim=1)
        theta_mean = self.fc(mean)
        theta = theta_mean
        theta = 2 * self.sigmoid(theta) - 1
        return theta

    def forward(self, x): # [n_voxels, n_points, n_channels]
        assert x.shape[2] == self.channels
        theta = self.get_relu_coefs(x)  # [n_voxels, n_channels * 2 * k]
        relu_coefs = theta.view(-1, self.channels, 2 * self.k) * self.lambdas + self.init_v

        # BxCxL -> LxCxBx1
        x_perm = x.permute(1,0,2).unsqueeze(-1)
        output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
        # LxCxBx2 -> BxCxL
        result = torch.max(output, dim=-1)[0].transpose(0,1)
        return result


class DAModule(nn.Module):
    def __init__(self, n_points, n_channels, reduction_rate=4, channel_att=True,
                 point_att=True, task_aware=True, pool_types=['max', 'mean']):
        super(DAModule, self).__init__()
        self.name = 'DAModule'
        self.use_channel_att = channel_att
        self.use_point_att = point_att
        self.use_task_aware = task_aware
        print('channel_att, point_att, task_aware',channel_att,point_att, task_aware)
        if point_att:
            self.point_att = PCAttention(n_points, reduction_rate=reduction_rate, activation=nn.ReLU(),
                                         pool_types=pool_types)
        if channel_att:
            self.channel_att = PCAttention(n_channels, reduction_rate=reduction_rate, activation=nn.ReLU(),
                                           channel_mean=True, pool_types=pool_types)
        if task_aware:
            self.task_aware = TaskAware(n_channels, reduction_rate=reduction_rate, pool_types=pool_types)

    def forward(self, x):  # shape [n_voxels, n_points, n_channels]

        point_weight = self.point_att(x.permute(0, 2, 1)) \
            if self.use_point_att else torch.tensor(1.)  # [n_voxels, n_points, 1]
        channel_weight = self.channel_att(x).permute(0, 2, 1) \
            if self.use_channel_att else torch.tensor(1.)  # [n_voxels, 1, n_channels]

        if torch.any(point_weight != 1.) or torch.any(channel_weight != 1.):
            beta = torch.mul(channel_weight, point_weight)
            attention = beta
            x = x * attention  # shape [n_voxels, n_points, n_channels]

        if self.use_task_aware:
            x = self.task_aware(x)  # shape [n_voxels, n_points, n_channels]

        return x


class AttentionModule(nn.Module):
    def __init__(self, n_points, n_channels, reduction_rate=4):
        super(AttentionModule, self).__init__()
        self.name = 'PFNLayerDA'
        self.point_att = PCAttention(n_points, reduction_rate=reduction_rate, activation=nn.ReLU())
        self.channel_att = PCAttention(n_channels, reduction_rate=reduction_rate, activation=nn.ReLU(),
                                       channel_mean=True)
        self.task_aware = TaskAware(n_channels, reduction_rate=reduction_rate)

    def forward(self, x):  # shape [n_voxels, n_points, n_channels]
        point_weight = self.point_att(x.permute(0, 2, 1))  # [n_voxels, n_points, 1]
        channel_weight = self.channel_att(x).permute(0, 2, 1)  # [n_voxels, 1, n_channels]
        beta = torch.mul(channel_weight, point_weight)
        attention = torch.sigmoid(beta)
        x = x * attention  # shape [n_voxels, n_points, n_channels]
        out = self.task_aware(x).permute(0, 2, 1)  # shape [n_voxels, n_points, n_channels]
        return out


class DAFeature(nn.Module):
    def __init__(self, dim_channels=9, dim_points=100, reduction_rate=8, boost_channels=64,
                 residual=False, relu=True, channel_att=True, point_att=True, task_aware=True,
                 pool_types=['max', 'mean']):
        super(DAFeature, self).__init__()
        self.residual = residual
        self.att_module1 = DAModule(n_points=dim_points, n_channels=dim_channels,
                                    reduction_rate=reduction_rate,
                                    channel_att=channel_att,
                                    point_att=point_att,
                                    task_aware=task_aware,
                                    pool_types=pool_types)  # linear last

        if residual:
            self.fc1 = nn.Sequential(
                nn.Linear(dim_channels, boost_channels),
                nn.ReLU(inplace=True) if relu else nn.Identity(),
            )
        else:
            self.fc1 = nn.Sequential(
                nn.Linear(dim_channels * 2, boost_channels),
                nn.ReLU(inplace=True) if relu else nn.Identity(),
            )

    def forward(self, x):  # [n_voxels, n_points, n_channels]
        out1 = self.att_module1(x)
        if self.residual:
            out1 = out1 + x
        else:
            out1 = torch.cat([out1, x], dim=2)  # [n_voxels, n_points, 2*n_channels]
        out1 = self.fc1(out1)  # Linear last

        return out1


class DAFELayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False,
                 dim_channels=9,
                 dim_points=100,
                 reduction_rate=8,
                 boost_channels=64,
                 channel_att=True,
                 point_att=True,
                 task_aware=True,
                 pool_types=['max', 'mean']
                 ):
        """
        Dynamic Attention Feature Extraction Layer.
        The Dynamic Attention Feature Extraction could be composed of a series of these layers, but the PointPillars paper results only
        used a single PFNLayer. This layer performs a similar role as second.pytorch.voxelnet.VFELayer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """

        super().__init__()
        self.name = 'DAFELayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        if use_norm:
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)

        self.extractor = DAFeature(dim_channels=dim_channels,
                                   dim_points=dim_points,
                                   reduction_rate=4,
                                   boost_channels=boost_channels,
                                   residual=False,
                                   channel_att=channel_att,
                                   point_att=point_att,
                                   task_aware=task_aware,
                                   pool_types=pool_types
                                   )  # Linear first
        self.extractor2 = DAFeature(dim_channels=boost_channels,
                                    dim_points=dim_points,
                                    reduction_rate=reduction_rate,
                                    boost_channels=boost_channels,
                                    residual=True,
                                    channel_att=channel_att,
                                    point_att=point_att,
                                    task_aware=task_aware,
                                    pool_types=pool_types
                                    # relu=False
                                    )
        self.linear = Linear(in_channels, self.units)
        self.norm = BatchNorm1d(self.units)

    def forward(self, inputs):
        n_voxel, n_points, n_channels = inputs.shape
        x = self.extractor(inputs)
        x = self.extractor2(x)
        x = self.linear(x)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x = F.relu(x)

        x_max = torch.max(x, dim=1, keepdim=True)[0]
        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        """
        Pillar Feature Net Layer.
        The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
        used a single PFNLayer. This layer performs a similar role as second.pytorch.voxelnet.VFELayer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """

        super().__init__()
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        if use_norm:
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)

        self.linear = Linear(in_channels, self.units)
        self.norm = BatchNorm1d(self.units)

    def forward(self, inputs):
        n_voxel, n_points, n_channels = inputs.shape
        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x = F.relu(x)

        x_max = torch.max(x, dim=1, keepdim=True)[0]
        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


@register_vfe
class PillarFeatureDANet(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=(64,),
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1),
                 boost_channel_dim=64,
                 reduction_rate=8,
                 num_point_per_voxel=100,
                 channel_att=True,
                 point_att=True,
                 task_aware_att=True,
                 pool_types=['max', 'mean']
                 ):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """

        super().__init__()
        self.name = 'PillarFeatureDANet'
        assert len(num_filters) > 0
        num_input_features += 5
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance
        print('pool_typesssssss',pool_types)

        # Create PillarFeatureNetOld layers
        num_filters = [boost_channel_dim] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(DAFELayer(in_filters, out_filters, use_norm, last_layer=last_layer,
                                        dim_channels=num_input_features,
                                        dim_points=num_point_per_voxel,
                                        reduction_rate=reduction_rate,
                                        boost_channels=boost_channel_dim,
                                        channel_att=channel_att,
                                        point_att=point_att,
                                        task_aware=task_aware_att,
                                        pool_types=pool_types)
                              )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]

    def forward(self, features, num_voxels, coors):
        device = features.device

        dtype = features.dtype
        # Find distance of x, y, and z from cluster center
        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center
        f_center = torch.zeros_like(features[:, :, :2])
        f_center[:, :, 0] = features[:, :, 0] - (coors[:, 3].to(dtype).unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = features[:, :, 1] - (coors[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset)

        # Combine together feature decorations
        features_ls = [features, f_cluster, f_center]
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        features = torch.cat(features_ls, dim=-1)

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask
        # Forward pass through PFNLayers
        for pfn in self.pfn_layers:
            features = pfn(features)

        return features.squeeze()


@register_vfe
class PillarFeatureNet(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=(64,),
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1),
                 ):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """

        super().__init__()
        self.name = 'PillarFeatureNetOld'
        assert len(num_filters) > 0
        num_input_features += 5
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        # Create PillarFeatureNetOld layers
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(PFNLayer(in_filters, out_filters, use_norm, last_layer=last_layer))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]

    def forward(self, features, num_voxels, coors):
        device = features.device

        dtype = features.dtype
        # Find distance of x, y, and z from cluster center
        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center
        f_center = torch.zeros_like(features[:, :, :2])
        f_center[:, :, 0] = features[:, :, 0] - (coors[:, 3].to(dtype).unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = features[:, :, 1] - (coors[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset)

        # Combine together feature decorations
        features_ls = [features, f_cluster, f_center]
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        features = torch.cat(features_ls, dim=-1)

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        # Forward pass through PFNLayers
        for pfn in self.pfn_layers:
            features = pfn(features)

        return features.squeeze()


@register_middle
class PointPillarsScatter(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=64,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='SpMiddle2K'):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """

        super().__init__()
        self.name = 'PointPillarsScatter'
        self.output_shape = output_shape
        self.ny = output_shape[2]
        self.nx = output_shape[3]
        self.nchannels = num_input_features

    def forward(self, voxel_features, coords, batch_size):
        # batch_canvas will be the final output.
        # print("voxel_features", voxel_features.shape)
        # print("coords", coords.shape)
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(
                self.nchannels,
                self.nx * self.ny,
                dtype=voxel_features.dtype,
                device=voxel_features.device)

            # Only include non-empty pillars
            batch_mask = coords[:, 0] == batch_itt
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(batch_size, self.nchannels, self.ny,
                                         self.nx)
        return batch_canvas
