from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch
import numpy as np


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight,top_k):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight
        self.top_k = top_k

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        losses = []

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            loss_each = self.criterion(
                heatmap_pred.mul(target_weight[:, idx]),
                heatmap_gt.mul(target_weight[:, idx])
            )
            loss_copy = loss_each.cpu()
            loss_copy = loss_copy.detach().numpy()
            losses.append(loss_copy)
        losses = np.array(losses)
        # print(losses)
        losses = torch.from_numpy(losses)
        # print(losses.shape)
        # print(losses)

        # for idx in range(num_joints):
        #     sub_loss = losses[idx]
        topk_val, topk_idx = torch.topk(losses, k=self.top_k, dim=0, sorted=False, out=None)
        tmp_loss = torch.gather(losses, 0, topk_idx)
        ohkm_loss = torch.sum(tmp_loss) / self.top_k
        ohkm_loss = torch.tensor(ohkm_loss,requires_grad=True)
        # print(ohkm_loss)

        return ohkm_loss

# def __init__(self, use_target_weight,top_k):
#     super(JointsMSELoss, self).__init__()
#     self.criterion = nn.MSELoss(size_average=True)
#     self.use_target_weight = use_target_weight
#     self.top_k = top_k
#
# def pre(self, output, target, target_weight):
#     batch_size = output.size(0)
#     num_joints = output.size(1)
#     heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
#     heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
#     ohkm_loss = 0
#     losses = []
#
#     for idx in range(num_joints):
#         heatmap_pred = heatmaps_pred[idx].squeeze()
#         heatmap_gt = heatmaps_gt[idx].squeeze()
#         loss_each = self.criterion(
#             heatmap_pred.mul(target_weight[:, idx]),
#             heatmap_gt.mul(target_weight[:, idx])
#         )
#         losses.append(loss_each)
#     return losses
#
# def ohkm(self,losses,top_k)
#     for idx in range(num_joints):
#         sub_loss = losses[idx]
#         topk_val, topk_idx = torch.topk(sub_loss, k=8, dim=0, sorted=False)
#         tmp_loss = torch.gather(sub_loss, 0, topk_idx)
#         ohkm_loss += torch.sum(tmp_loss) / self.top_k
#     return ohkm_loss / num_joints
#
# def forward(self):
#
#
#
#
#     return ohkm_loss / num_joints
