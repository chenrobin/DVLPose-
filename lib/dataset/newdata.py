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
    def __init__(self, img_dir, img_set, is_train, transform=None):
        self.img_dir = img_dir
        self.image_set = img_set
        self.anno = self._get_anno()
        self.transform = transform


    def _get_anno(self):

        vanns = []
        file_path = os.path.join(self.img_dir, self.image_set)
        file_path = Path(file_path)
        json_paths = tqdm(list(file_path.glob('*.json')))
        #deal each video
        for vid,  vann_path in enumerate(json_paths):
            with vann_path.open() as f:
                vann = json.load(f)

            vanns.append(vann)
            # if vid == 1:
            #     break
        return vanns

    def __getitem__(self, idx):
        db = copy.deepcopy(self.anno[idx])
        return db


    def __len__(self,):
        return len(self.anno)

    def evaluate(self, cfg, preds, output_dir, all_boxes, img_path,
                 *args, **kwargs):
        preds = preds[:, :, 0:2]
        pos_pred_src = np.transpose(preds, [1, 2, 0])#(17*3*21345)

        if output_dir:
            gt_file = Path(os.path.join(output_dir, 'valid2.json'))
        with gt_file.open() as f:
            metas = json.load(f)


        threshold = 0.5
        jnt_vis = []
        pos_gt_src = []
        ws = []
        hs = []
        wh = []
        hh = []
        SC_BIAS = 0.8*0.75

        for n in range(len(metas)):
            jnt_vis.append(np.array(metas[n]['joints_vis'][0])[0:17,1])
            pos_gt_src.append(np.array(metas[n]['joints'][0])[:,0:2])
            # ws.append(np.array(metas[n]['weight']))
            # hs.append(np.array(metas[n]['height']))
            wh.append(np.array(metas[n]['bbox_head'][2]))
            hh.append(np.array(metas[n]['bbox_head'][2]))


        pos_gt_src = np.array(pos_gt_src)
        jnt_vis = np.array(jnt_vis)
        whead = np.array(wh)
        hhead = np.array(hh)
        headsize = np.sqrt(whead**2+hhead**2)#(21348,1)
        headsize = headsize*SC_BIAS
        # weight = np.array(ws)
        # height = np.array(hs)
        # area = np.multiply(weight, height)#(21345,)
        # area = area[:,0]
        pos_gt_src = np.transpose(pos_gt_src,[1,2,0])#(17*3*21345)
        jnt_vis = np.transpose(jnt_vis,[1,0])#(17*21345)


        uv_error = pos_pred_src - pos_gt_src#(17*3*21345)
        uv_err = np.linalg.norm(uv_error, axis=1)**2#(17*21345)
        dis_sum = np.sum(uv_err,axis = 0)#(21345,)
        # print(dis_sum.shape)
        scale = np.multiply(headsize, np.ones((len(uv_err), 1)))
        scaled_uv_err = np.divide(uv_err, scale)
        scaled_uv_err = np.multiply(scaled_uv_err, jnt_vis)

        jnt_count = np.sum(jnt_vis, axis=1)
        less_than_threshold = np.multiply((scaled_uv_err <= 0.8),
                                          jnt_vis)
        PCKh = np.divide(100.*np.sum(less_than_threshold, axis=1), jnt_count)

        jnt_ratio = jnt_count / 21348


        # oks = np.exp(-dis_sum/2/area)
        # # print(oks.shape)
        #
        # ap = np.arange(0.5, 1, 0.05)
        # # ap_s = np.sum(oks >= threshold)/len(oks)
        # AP = np.zeros(len(ap))

        # oks_part = np.exp(-uv_err/2/area)
        # ap_p = np.sum(oks_part >= 0.5)/oks_part.shape[1]

        # for r in range(len(ap)):
        #     threshold = ap[r]
        #     # print(np.sum(oks >= threshold))
        #     # print(len(oks))
        #     ap_s = np.sum(oks >= threshold)/len(oks)
        #     AP[r] = ap_s
        #
        #
        # name_value = [
        #     ('AP.5', AP[0]),
        #     ('AP.55', AP[1]),
        #     ('AP.6', AP[2]),
        #     ('AP.65', AP[3]),
        #     ('AP.7', AP[4]),
        #     ('AP.75', AP[5]),
        #     ('AP.8', AP[6]),
        #     ('AP.85', AP[7]),
        #     ('AP.9', AP[8]),
        #     ('AP.95', AP[9]),
        # ]
        name_value = [
            ('Head', PCKh[2]),
            ('Shoulder', 0.5 * (PCKh[5] + PCKh[6])),
            ('Elbow', 0.5 * (PCKh[7] + PCKh[8])),
            ('Wrist', 0.5 * (PCKh[9] + PCKh[10])),
            ('Hip', 0.5 * (PCKh[11] + PCKh[12])),
            ('Knee', 0.5 * (PCKh[13] + PCKh[14])),
            ('Ankle', 0.5 * (PCKh[15] + PCKh[16])),
            ('Mean', np.sum(PCKh * jnt_ratio)),
        ]
        name_value = OrderedDict(name_value)

        return name_value, name_value['Mean']
