#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM

Modified by 
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@Time: 2020/3/9 9:32 PM
"""

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from util import to_cuda

VERY_SMALL_NUMBER = 1e-12
INF = 1e20


class GlobalAttention(nn.Module):
    def __init__(self, input_size, k):
        super(GlobalAttention, self).__init__()
        self.k = k
        self.weight = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(1, 1, input_size)))

    def forward(self, x):  # B x N x C
        B, N, C = x.size(0), x.size(1), x.size(2)
        att = F.softmax(torch.sum(x * self.weight, dim=-1) / math.sqrt(C), dim=-1)  # B x N
        _, sort_idx = torch.sort(att, dim=1)  # B x N
        # sort_idx = sort_idx.detach()
        knn_val = torch.zeros(B, N, self.k, dtype=x.dtype, device=x.device, requires_grad=True)
        knn_idx = torch.zeros(B, N, self.k, dtype=sort_idx.dtype, device=x.device)
        for b in range(B):
            for i in range(N):
                if i < self.k / 2:
                    knn_idx[b, sort_idx[b, i], :] = sort_idx[b, 0:self.k]
                elif i < N - self.k / 2:
                    knn_idx[b, sort_idx[b, i], :] = sort_idx[b, i - self.k // 2: i + self.k // 2]
                else:
                    knn_idx[b, sort_idx[b, i], :] = sort_idx[b, -self.k:]
                # knn_val[b, sort_idx[b, i], :] = att[b, knn_idx[b, sort_idx[b, i], :]]
        index = torch.gather(input=knn_idx, dim=1, index=sort_idx[..., None].expand(-1, -1, self.k))
        src = torch.gather(input=att[..., None].expand(-1, -1, self.k), dim=1, index=index)
        knn_val = torch.scatter(input=knn_val, dim=1, index=sort_idx[..., None].expand(-1, -1, self.k), src=src)
        # knn_val.requires_grad = True
        return knn_val, knn_idx


class GraphLearner(nn.Module):
    def __init__(self, input_size, hidden_size, num_pers=16, metric_type='attention'):
        super(GraphLearner, self).__init__()
        self.metric_type = metric_type
        if metric_type == 'attention':
            self.linear_sims = nn.ModuleList([nn.Linear(input_size, hidden_size, bias=False) for _ in range(num_pers)])
            print('[ Multi-perspective {} GraphLearner: {} ]'.format(metric_type, num_pers))

        elif metric_type == 'weighted_cosine':
            self.weight_tensor = torch.Tensor(num_pers, input_size)
            self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))
            print('[ Multi-perspective {} GraphLearner: {} ]'.format(metric_type, num_pers))

        elif metric_type == 'gat_attention':
            self.linear_sims1 = nn.ModuleList([nn.Linear(input_size, 1, bias=False) for _ in range(num_pers)])
            self.linear_sims2 = nn.ModuleList([nn.Linear(input_size, 1, bias=False) for _ in range(num_pers)])
            self.leakyrelu = nn.LeakyReLU(0.2)
            print('[ GAT_Attention GraphLearner]')

        elif metric_type == 'kernel':
            self.precision_inv_dis = nn.Parameter(torch.Tensor(1, 1))
            self.precision_inv_dis.data.uniform_(0, 1.0)
            self.weight = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(input_size, hidden_size)))

        elif metric_type == 'transformer':
            self.lin_wq = nn.Linear(input_size, hidden_size, bias=False)
            self.lin_wk = nn.Linear(input_size, hidden_size, bias=False)

        elif metric_type == 'Gaussian_Mahalanobis':
            self.lin = nn.Linear(input_size, hidden_size, bias=False)

        else:
            raise ValueError('Unknown metric_type: {}'.format(metric_type))

        print('[ Graph Learner metric type: {} ]'.format(metric_type))

    def forward(self, context):
        """
        Parameters
        :context, (batch_size, ctx_size, dim)

        Returns
        :similarity, (batch_size, ctx_size, ctx_size)
        """
        if self.metric_type == 'attention':
            similarity = 0
            for _ in range(len(self.linear_sims)):
                context_fc = torch.relu(self.linear_sims[_](context))
                similarity += torch.matmul(context_fc, context_fc.transpose(-1, -2))
            similarity /= len(self.linear_sims)

        elif self.metric_type == 'weighted_cosine':
            expand_weight_tensor = self.weight_tensor.unsqueeze(1)
            if len(context.shape) == 3:
                expand_weight_tensor = expand_weight_tensor.unsqueeze(1)
            context_fc = context.unsqueeze(0) * expand_weight_tensor
            context_norm = F.normalize(context_fc, p=2, dim=-1)
            similarity = torch.matmul(context_norm, context_norm.transpose(-1, -2)).mean(0)
            similarity = F.relu(similarity)

        elif self.metric_type == 'transformer':
            Q = self.lin_wq(context)
            K = self.lin_wk(context)
            similarity = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(Q.shape[-1])
            similarity = F.softmax(similarity, dim=-1)

        elif self.metric_type == 'gat_attention':
            similarity = []
            for _ in range(len(self.linear_sims1)):
                a_input1 = self.linear_sims1[_](context)
                a_input2 = self.linear_sims2[_](context)
                similarity.append(self.leakyrelu(a_input1 + a_input2.transpose(-1, -2)))
            similarity = torch.mean(torch.stack(similarity, 0), 0)

        elif self.metric_type == 'Gaussian_Mahalanobis':
            x = self.lin(context)
            inner = torch.matmul(x, x.transpose(-1, -2))
            norm2 = torch.sum(x ** 2, dim=-1, keepdim=True)
            similarity = -norm2 + 2 * inner - norm2.transpose(-1, -2)  # negative_pairwise_distance
            similarity = F.softmax(similarity, dim=-1)

        return similarity


def knn(x, k, metric='Euclidean'):
    if metric == 'Euclidean':
        inner = torch.matmul(x.transpose(2, 1), x)
        norm2 = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_similarity = -norm2 + 2 * inner - norm2.transpose(2, 1)  # negative_pairwise_distance
    elif metric == 'Cosine':
        x_normed = F.normalize(x, p=2, dim=1)
        pairwise_similarity = torch.matmul(x_normed.transpose(2, 1), x_normed)
    else:
        exit('Unknown distance metric!')
        return None

    idx = pairwise_similarity.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, weight=None, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    num_dims = x.size(1)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)

    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    if weight is not None:
        weight = weight.view(-1, 1)

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    if weight is not None:
        feature = torch.mul(feature, weight)
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  # (batch_size, 2*num_dims, num_points, k)


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class DGCNN_cls(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN_cls, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k)     # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)                       # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)

        x = self.conv5(x)                       # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)              # (batch_size, emb_dims*2)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2) # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x = self.linear3(x)                                             # (batch_size, 256) -> (batch_size, output_channels)
        
        return x


class Transform_Net(nn.Module):
    def __init__(self, args):
        super(Transform_Net, self).__init__()
        self.args = args
        self.k = 3

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3 * 3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)            # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x


class DGCNN_partseg(nn.Module):
    def __init__(self, args, seg_num_all):
        super(DGCNN_partseg, self).__init__()
        self.args = args
        self.seg_num_all = seg_num_all
        self.k = args.k
        self.transform_net = Transform_Net(args)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(1280, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn10,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)

    def forward(self, x, l):
        batch_size = x.size(0)
        num_points = x.size(2)

        x0 = get_graph_feature(x, k=self.k)     # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        t = self.transform_net(x0)              # (batch_size, 3, 3)
        x = x.transpose(2, 1)                   # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)                     # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1)                   # (batch_size, num_points, 3) -> (batch_size, 3, num_points)

        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        l = l.view(batch_size, -1, 1)           # (batch_size, num_categoties, 1)
        l = self.conv7(l)                       # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)

        x = torch.cat((x, l), dim=1)            # (batch_size, 1088, 1)
        x = x.repeat(1, 1, num_points)          # (batch_size, 1088, num_points)

        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1088+64*3, num_points)

        x = self.conv8(x)                       # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        x = self.dp2(x)
        x = self.conv10(x)                      # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        x = self.conv11(x)                      # (batch_size, 256, num_points) -> (batch_size, seg_num_all, num_points)
        
        return x


class DGCNN_semseg(nn.Module):
    def __init__(self, args):
        super(DGCNN_semseg, self).__init__()
        self.args = args
        self.k = args.k
        self.num_features = args.num_features

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        self.conv1 = nn.Sequential(nn.Conv2d(18, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Conv1d(256, 13, kernel_size=1, bias=False)

        self.gsl1 = GraphLearner(self.num_features, 64, num_pers=8, metric_type='weighted_cosine')  # Gaussian_Mahalanobis
        # self.ga1 = GlobalAttention(self.num_features, self.k)
        # self.ga2 = GlobalAttention(64, self.k)
        # self.ga3 = GlobalAttention(64, self.k)

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        adj = self.gsl1(x.transpose(2, 1))
        knn_val, knn_idx = adj.topk(k=self.k, dim=-1)

        # knn_val, knn_idx = self.ga1(x.transpose(2, 1))

        x = get_graph_feature(x, k=self.k, weight=knn_val, idx=knn_idx)   # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        # knn_val, knn_idx = self.ga1(x1.transpose(2, 1))

        x = get_graph_feature(x1, k=self.k, weight=knn_val, idx=knn_idx)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        # knn_val, knn_idx = self.ga1(x2.transpose(2, 1))

        x = get_graph_feature(x2, k=self.k, weight=knn_val, idx=knn_idx)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)          # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1024+64*3, num_points)

        x = self.conv7(x)                       # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)                       # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 13, num_points)
        
        return x