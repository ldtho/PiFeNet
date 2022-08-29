import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torch import einsum
# from second.pytorch.models.BoTnet_CCA import CCA,DyReLUB
from collections import OrderedDict
from einops import rearrange
from torch.nn import init
import math
from second.pytorch.utils import SwishImplementation, Swish, SeparableConvBlock, MemoryEfficientSwish, \
    PositionEmbeddingSine, PositionEmbeddingLearned, inverse_sigmoid, MLP

class MiniBiFPN(nn.Module):  # input size: batch, C, width, length [1,64,400,600]
    def __init__(self,
                 num_blocks=[3, 4, 6],
                 mid_planes=[64, 128, 128],
                 num_class=1000,
                 resolution=[400, 400],
                 strides=[1, 2, 2],
                 upsample_strides=[2, 4, 4],
                 num_upsample_filters=[128, 128, 128],
                 num_input_filters=128,
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=False,
                 box_code_size=7,
                 num_direction_bins=2,
                 onnx_export=False,
                 attention=True,
                 ):
        super(MiniBiFPN, self).__init__()
        self.resolution = list(resolution)
        self.in_planes = num_input_filters
        self._num_class = num_class
        self._box_code_size = box_code_size
        self._num_direction_bins = num_direction_bins
        self._num_anchor_per_loc = num_anchor_per_loc
        self._use_direction_classifier = use_direction_classifier
        self.act = MemoryEfficientSwish() if not onnx_export else Swish()
        self.layer1 = [
            nn.Conv2d(num_input_filters, mid_planes[0], 3, stride=strides[0], padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_planes[0], eps=1e-3, momentum=0.01),
            self.act
        ]
        for i in range(num_blocks[0]):
            # conv = nn.Conv2d if i < num_blocks[0] -1 else DeformableConv2d
            conv = nn.Conv2d
            self.layer1 += [conv(mid_planes[0], mid_planes[0], 3, 1, padding=1, bias=False),
                            nn.BatchNorm2d(mid_planes[0], eps=1e-3, momentum=0.01),
                            self.act]
        self.layer1 = nn.Sequential(*self.layer1)

        upsample_strides = list(map(float, upsample_strides))
        print('upsample_strides',upsample_strides)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(mid_planes[1],
                               num_upsample_filters[0],
                               int(upsample_strides[0]),
                               stride=int(upsample_strides[0]), bias=False
                               ) if upsample_strides[0] >= 1 else
            nn.Conv2d(mid_planes[1],
                      num_upsample_filters[0],
                      kernel_size=int(1/upsample_strides[0]),
                      stride=int(1/upsample_strides[0]),
                      bias=False),
            nn.BatchNorm2d(num_upsample_filters[0], eps=1e-3, momentum=0.01),
            self.act
        )

        self.layer2 = [
            nn.Conv2d(mid_planes[0], mid_planes[1], 3, stride=strides[1], padding=1, bias=False),
            nn.BatchNorm2d(mid_planes[1], eps=1e-3, momentum=0.01),
            self.act
        ]
        for i in range(num_blocks[1]):
            conv = nn.Conv2d
            self.layer2 += [conv(mid_planes[1], mid_planes[1], 3, 1, padding=1, bias=False),
                            nn.BatchNorm2d(mid_planes[1], eps=1e-3, momentum=0.01),
                            self.act]
        self.layer2 = nn.Sequential(*self.layer2)

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(mid_planes[1],
                               num_upsample_filters[1],
                               int(upsample_strides[1]),
                               stride=int(upsample_strides[1]), bias=False
                               ) if upsample_strides[1] >= 1 else
            nn.Conv2d(mid_planes[1],
                      num_upsample_filters[1],
                      kernel_size=int(1/upsample_strides[1]),
                      stride=int(1/upsample_strides[1]),
                      bias=False),
            nn.BatchNorm2d(num_upsample_filters[1], eps=1e-3, momentum=0.01),
            self.act
        )

        self.layer3 = [
            nn.Conv2d(mid_planes[1], mid_planes[2], 3, stride=strides[2], padding=1, bias=False),
            nn.BatchNorm2d(mid_planes[2], eps=1e-3, momentum=0.01),
            self.act
        ]
        for i in range(num_blocks[2]):
            conv = nn.Conv2d
            self.layer3 += [conv(mid_planes[2], mid_planes[2], 3, 1, padding=1, bias=False),
                            nn.BatchNorm2d(mid_planes[2], eps=1e-3, momentum=0.01),
                            self.act]

        self.layer3 = nn.Sequential(*self.layer3)
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(mid_planes[1],
                               num_upsample_filters[2],
                               int(upsample_strides[2]),
                               stride=int(upsample_strides[2]), bias=False
                               ) if upsample_strides[2] >= 1 else
            nn.Conv2d(mid_planes[1],
                      num_upsample_filters[2],
                      kernel_size=int(1/upsample_strides[2]),
                      stride=int(1/upsample_strides[2]),
                      bias=False),
            nn.BatchNorm2d(num_upsample_filters[2], eps=1e-3, momentum=0.01),
            self.act
        )

        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        self.conv_cls = nn.Conv2d(sum(num_upsample_filters), num_cls, kernel_size=1)
        self.conv_box = nn.Conv2d(sum(num_upsample_filters), num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(sum(num_upsample_filters), num_anchor_per_loc * num_direction_bins,
                                          kernel_size=1)
        self.p1_down_channel = nn.Sequential(nn.Conv2d(mid_planes[0], mid_planes[1], 1),
                                             nn.BatchNorm2d(mid_planes[1], momentum=0.01, eps=1e-3))
        self.p2_down_channel = nn.Sequential(nn.Conv2d(mid_planes[1], mid_planes[1], 1),
                                             nn.BatchNorm2d(mid_planes[1], momentum=0.01, eps=1e-3))
        self.p3_down_channel = nn.Sequential(nn.Conv2d(mid_planes[2], mid_planes[1], 1),
                                             nn.BatchNorm2d(mid_planes[1], momentum=0.01, eps=1e-3))
        self.conv2_up = SeparableConvBlock(mid_planes[1], norm=True, activation=False, onnx_export=onnx_export)
        self.conv1_up = SeparableConvBlock(mid_planes[1], norm=True, activation=True,
                                           onnx_export=onnx_export)  # use act

        self.p1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p2_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p2_downsample = nn.MaxPool2d((2, 2))
        self.p3_downsample = nn.MaxPool2d((2, 2))

        self.conv2_down = SeparableConvBlock(mid_planes[1], norm=True, activation=True,
                                             onnx_export=onnx_export)  # use act
        self.conv3_down = SeparableConvBlock(mid_planes[1], norm=True, activation=True,
                                             onnx_export=onnx_export)  # use act
        # height, width = int(resolution[0] / strides[0]), int(resolution[1] / strides[0])
        # print('height, width', height, width)
        self.relu = nn.ReLU()
        self.attention = attention
        if attention:
            self.epsilon = 1e-5
            self.p2_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
            self.p1_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
            self.p2_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
            self.p3_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

    def _forward(self, p1, p2, p3):
        p2_up = self.conv2_up(self.act(p2 + self.p2_upsample(p3)))
        p1_out = self.conv1_up(self.act(p1 + self.p1_upsample(p2_up)))
        p2_out = self.conv2_down(self.act(p2 + p2_up + self.p2_downsample(p1_out)))
        p3_out = self.conv3_down(self.act(p3 + self.p3_downsample(p2_out)))

        return p1_out, p2_out, p3_out

    def _forward_fast_attention(self, p1, p2, p3):
        p2_w1 = self.relu(self.p2_w1)
        weight = p2_w1 / (torch.sum(p2_w1, dim=0) + self.epsilon)
        p2_up = self.conv2_up(self.act(weight[0] * p2 + weight[1] * self.p2_upsample(p3)))
        p1_w1 = self.relu(self.p1_w1)
        weight = p1_w1 / (torch.sum(p1_w1, dim=0) + self.epsilon)

        p1_out = self.conv1_up(self.act(weight[0] * p1 + weight[1] * self.p1_upsample(p2_up)))

        p2_w2 = self.relu(self.p2_w2)
        weight = p2_w2 / (torch.sum(p2_w2, dim=0) + self.epsilon)
        p2_out = self.conv2_down(self.act(weight[0] * p2 + weight[1] * p2_up + weight[2] * self.p2_downsample(p1_out)))

        p3_w2 = self.relu(self.p3_w2)
        weight = p3_w2 / (torch.sum(p3_w2, dim=0) + self.epsilon)
        p3_out = self.conv3_down(self.act(weight[0] * p3 + weight[1] * self.p3_downsample(p2_out)))

        return p1_out, p2_out, p3_out

    def forward(self, x):
        x = self.layer1(x)
        p1 = self.p1_down_channel(x)
        x = self.layer2(x)
        p2 = self.p2_down_channel(x)
        x = self.layer3(x)
        p3 = self.p3_down_channel(x)

        if self.attention:
            p1_out, p2_out, p3_out = self._forward_fast_attention(p1, p2, p3)
        else:
            p1_out, p2_out, p3_out = self._forward(p1, p2, p3)

        up1 = self.deconv1(p1_out)
        up2 = self.deconv2(p2_out)
        up3 = self.deconv3(p3_out)

        x = torch.cat([up1, up2, up3], dim=1)
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        C, H, W = box_preds.shape[1:]
        box_preds = box_preds.view(-1, self._num_anchor_per_loc,
                                   self._box_code_size, H, W).permute(0, 1, 3, 4, 2).contiguous()
        cls_preds = cls_preds.view(-1, self._num_anchor_per_loc,
                                   self._num_class, H, W).permute(0, 1, 3, 4, 2).contiguous()
        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.view(-1, self._num_anchor_per_loc, self._num_direction_bins,
                                               H, W).permute(0, 1, 3, 4, 2).contiguous()
            ret_dict["dir_cls_preds"] = dir_cls_preds
        return ret_dict


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_params(m):
    if type(m) in {
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
        nn.Linear,
    }:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
            bound = 1 / math.sqrt(fan_out)
            nn.init.normal_(m.bias, -bound, bound)
    elif type(m) in {
        nn.BatchNorm2d,
        nn.BatchNorm1d,
    }:
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)

#
# if __name__ == '__main__':
#     import torch.onnx
#
#     batch_size = 1
#     input_height = 256
#     input_width = 288
#     model = MiniBiFPN(
#         num_blocks=[3, 5, 5],
#         mid_planes=[64, 128, 256],
#         strides=[1, 2, 2],
#         upsample_strides=[1, 2, 4],
#         num_class=2,
#         resolution=[input_height, input_width],
#         num_upsample_filters=[128, 128, 128],
#         num_input_filters=64,
#         num_anchor_per_loc=2,
#         encode_background_as_zeros=True,
#         use_direction_classifier=False,
#         box_code_size=7,
#         attention=True
#     )
#     model.apply(init_params)
#     x = torch.randn(batch_size, 64, input_height, input_width, requires_grad=True)
#     preds = model(x)
#     print(model)
#     print(count_parameters(model))
#     print('preds[box_preds].shape',preds['box_preds'].shape)
#
#     exit()
#     model.eval()
#     torch.onnx.export(model, x, 'test_onnx.onnx', export_params=True, opset_version=12,
#                       input_names=['input'],  # the model's input names
#                       output_names=['output'],  # the model's output names
#                       dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
#                                     'output': {0: 'batch_size'}}
#                       )
#     print('preds[0].shape', preds[0].shape)
#     print('preds[1].shape', preds[1].shape)
#     exit()
