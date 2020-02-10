from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import pickle
import copy
from collections import defaultdict
from collections import OrderedDict

from scipy.io import loadmat
import json_tricks as json
import numpy as np
import cv2
import torch
from tqdm import tqdm
from pathlib import Path
from pprint import pprint
from itertools import combinations
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class PoseTrackDataset(Dataset):
    '''
    "keypoints": [
            0-"nose", 1-"head_bottom", 2-"head_top", 3-"left_ear", 4-"right_ear",
            5-"left_shoulder", 6-"right_shoulder", 7-"left_elbow", 8-"right_elbow",
            9-"left_wrist", 10-"right_wrist", 11-"left_hip", 12-"right_hip",
            13-"left_knee", 14-"right_knee", 15-"left_ankle", 16-"right_ankle"
        ]
    '''
    def __init__(self, img_dir, img_set, is_train, transform=None):
        self.img_dir = img_dir
        self.image_set = img_set
        self.pairs = [[0, 1], [0, 2], [0, 3], [0, 4],
                      [1, 5], [5, 7], [7, 9],
                      [1, 6], [6, 8], [8, 10],
                      [5, 6], [11, 12],
                      [5, 11], [11, 13], [13, 15],
                      [6, 12], [12, 14], [14, 16]]
        self.keypoints = []
        # self.oks_thre = cfg.TEST.OKS_THRE
        # self.in_vis_thre = cfg.TEST.IN_VIS_THRE
        self.aspect_ratio = 192 * 1.0 / 256
        self.target_type = 'gaussian'
        self.heatmap_size = [48, 64]
        self.image_size = [192, 256]
        self.sigma = 2
        self.num_joints = 17
        self.pixel_std = 200
        self.anno = self._get_anno()
        self.indexs = self._getindex(self.anno)
        self.transform = transform


    def _get_anno(self):
        """
        /media/cc/000DFF1D000F537C/posetrack/posetrack2017/posetrack_data/images
        /media/cc/000DFF1D000F537C/posetrack/posetrack_data/annotations/train
        """
        vanns = []
        file_path = os.path.join(self.img_dir, 'annotations', self.image_set)
        file_path = Path(file_path)
        json_paths = tqdm(list(file_path.glob('*.json')))
        #deal each video
        for vid,  vann_path in enumerate(json_paths):
            with vann_path.open() as f:
                vann = json.load(f)

            image_anno = vann['annotations']
            image_dict = vann['images']
            video_id = image_dict[0]['vid_id']
            fanns = []
            #deal each frame

            for i in range(len(image_dict)):
                if not image_dict[i]['is_labeled']:
                    continue
                image_name = image_dict[i]['file_name']
                image_id = image_dict[i]['frame_id']
                # print('len:',len(image_anno))
                fpanns = []
                #deal each person in frame
                for p in range(len(image_anno)):
                    if image_dict[i]['id'] != image_anno[p]['image_id']:
                        continue
                    per_s = []
                    bbox = []
                    score = []
                    pid = []
                    joint = []
                    joint_vis = []
                    center = []
                    bbox_head = []
                    if 'bbox' not in image_anno[p]:
                        x, y, w, h = -1, -1, -1, -1
                    else:
                        x, y, w, h = image_anno[p]['bbox'][0],\
                                     image_anno[p]['bbox'][1],\
                                     image_anno[p]['bbox'][2],\
                                     image_anno[p]['bbox'][3]
                    c = np.zeros((2), dtype=np.float32)
                    c[0] = x + w * 0.5
                    c[1] = y + h * 0.5
                    if w > self.aspect_ratio * h:
                        h = w * 1.0 / self.aspect_ratio
                    elif w < self.aspect_ratio * h:
                        w = h * self.aspect_ratio
                    scale = np.array(
                        [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
                        dtype=np.float32)
                    if c[0] != -1:
                        scale = scale * 1.25
                    xh = image_anno[p]['bbox_head'][0]
                    yh = image_anno[p]['bbox_head'][1]
                    wh = image_anno[p]['bbox_head'][2]
                    hh = image_anno[p]['bbox_head'][3]
                    bbox_head.append([xh,yh,wh,hh])
                    per_s.append(scale)
                    bbox.append([x,y,w,h])
                    center.append(c)
                    score.append(image_anno[p]['scores'])
                    pid.append(image_anno[p]['track_id'])
                    for j in range(len(image_anno[p]['keypoints'])):
                        if 3*j >= len(image_anno[p]['keypoints']):
                            break
                        joint.append([image_anno[p]['keypoints'][j*3],image_anno[p]['keypoints'][j*3+1]])
                        joint_vis.append(image_anno[p]['keypoints'][j*3+2])
                    #2019.2.24
                    joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
                    joints_3d_vis = np.zeros((self.num_joints,  3), dtype=np.float)
                    joints = np.array(joint)
                    joints_3d[:, 0:2] = np.array(joints[:, 0:2])
                    joints_3d_vis[:, 0] = joint_vis[:]
                    joints_3d_vis[:, 1] = joint_vis[:]
                    #2019.2.24
                    fpanns.append({
                    'per_s': per_s,
                    'person_id': pid,
                    'bbox': bbox,
                    'center': center,
                    'score': score,
                    'joint': joints_3d,
                    'joint_vis': joints_3d_vis,
                    'bbox_head': bbox_head,
                    })

                # print(fpanns)
                fanns.append({
                'image_name': image_name,
                'frame-person_ann': fpanns,
                })
            vanns.append(fanns)
            # if vid == 0:
            #     break
            # print(vanns)

        return vanns

    def affine_transform(self, pt, t):
        new_pt = np.array([pt[0], pt[1], 1.]).T
        new_pt = np.dot(t, new_pt)
        return new_pt[:2]

    def _getindex(self,vanns):
        index = []
        inds = []
        num_index = []
        id = 0
        for vid in range(len(vanns)):
            db_rec = copy.deepcopy(vanns[vid])
            flen = len(db_rec)
            for fid in range(flen):
                for pid in range(len(db_rec[fid]['frame-person_ann'])):
                    bbox = db_rec[fid]['frame-person_ann'][pid]['bbox'][0]
                    if bbox[0]<=0 or bbox[1]<=0 or bbox[2]<=0 or bbox[3]<=0:
                        continue
                    index.append([vid,fid,pid])
            index.sort(key = lambda x:x[2])
            if index == []:
                continue
            tag = index[0][2]
            for n in range(len(index)):
                if index[n][2] != tag:
                    inds.append([id,num_index])
                    id = id + 1
                    tag = index[n][2]
                    num_index = []
                    num_index.append(index[n])
                else:
                    num_index.append(index[n])
            inds.append([id,num_index])
            id = id + 1
            index = []
            num_index = []

        return inds

    def __getitem__(self, idx):
        db = copy.deepcopy(self.indexs[idx])
        # print('idx:',idx)
        metas = []
        inputs = []
        targets = []
        target_weights = []
        # print('id:',db[0])
        for i in range(len(db[1])):
            vdb = db[1][i]
            vid = vdb[0]
            fid = vdb[1]
            pid = vdb[2]
            # print('1:',[vid,fid,pid])
            db_rec = copy.deepcopy(self.anno[vid])
            image_file = os.path.join(self.img_dir, db_rec[fid]['image_name'])
            anno = db_rec[fid]['frame-person_ann']

            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            r = 0
            c = np.array(anno[pid]['center'][0])
            s = anno[pid]['per_s'][0]
            bbox = anno[pid]['bbox'][0]
            bbox_head = anno[pid]['bbox_head'][0]
            id = anno[pid]['person_id'][0]
            w = bbox[2]
            h = bbox[3]
            joints = anno[pid]['joint']
            joints_vis = anno[pid]['joint_vis']

            joints[:,0] = joints[:,0] - bbox[0]
            joints[:,1] = joints[:,1] - bbox[1]
            joints[joints<0]=0

            data = data_numpy[int(bbox[1]):int(bbox[1]+bbox[3]),int(bbox[0]):int(bbox[0]+bbox[2])]
            pts1 = np.float32([[0,0],[0,data.shape[0]],[data.shape[1],0]])
            pts2 = np.float32([[0,0],[0,256],[192,0]])
            trans =  cv2.getAffineTransform(pts1,pts2)

            if bbox[0]<=0 or bbox[1]<=0 or bbox[2]<=0 or bbox[3]<=0:
                return 0
            input = cv2.warpAffine(
                data,
                trans,
                (192, 256),
                flags=cv2.INTER_LINEAR)
            # print('2:',[vid,fid,pid])

            if self.transform:
                input = self.transform(input)
            # print(input.shape)

            for i in range(len(joints)):
                if joints_vis[i,0] > 0.0:
                    joints[i, 0:2] = self.affine_transform(joints[i, 0:2],trans)

            target, target_weight = self.generate_target(joints, joints_vis)

            target = torch.from_numpy(target)
            # logger.info('test target{}'.format(target.shape))
            target_weight = torch.from_numpy(target_weight)
            meta = {
                'image':image_file,
                'joints':joints,
                'joints_vis':joints_vis,
                'center':c,
                'scale':s,
                'rotation':r,
                'weight': w,
                'height': h,
                'bbox_head': bbox_head,
                'id': id,
            }
            inputs.append(input)
            targets.append(target)
            target_weights.append(target_weight)
            metas.append(meta)

        # print(meta)


        return inputs, targets, target_weights, metas


    def __len__(self,):
        return len(self.indexs)

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = np.asarray(self.image_size) / np.asarray(self.heatmap_size)
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight
