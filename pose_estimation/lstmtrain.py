
import argparse
import os
import time
import pprint
import logging

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import sys
import numpy as np
sys.path.append("/media/cc/00035E420005A4F5/exper/lstmpose/lib")
print(sys.path)
from pathlib import Path
import json

import numpy as np
from core.loss import JointsMSELoss
from utils.utils import create_logger
from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.vis import save_debug_images
from utils.utils import get_optimizer
from core.config import get_model_name
from utils.utils import save_checkpoint

from core.config import config
from core.config import update_config


import models.singleModel
import dataset.PDVideo
import dataset.newdata
import models.posemodel
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)
    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        type=int)

    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
#
#
# class LSTM(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.lstmcell = torch.nn.LSTMCell(
#             inpout_size = 256*192*3,
#             hidden_size = 256,
#         )
#     def forward(self,x):
#         output,(h_n,c_n) = self.lstmcell(x)
#         # output_last = h_n[-1,:,:]
#         return output,(h_n,c_n)

# class getPose(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.final_layer = nn.Conv2d(
#             in_channels=256,
#             out_channels=17,
#             kernel_size=1,
#             stride=1,
#             padding=0
#         )
#     def forward(self,x):
#         x = self.final_layer(x)
#         return x

# class lstmModel(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.lstm = nn.LSTMCell(17*64*48,64*48)
#         self.conv1 = nn.Conv2d(in_channels=1,
#         out_channels=17,
#         kernel_size=1,
#         stride=1,
#         padding=0)
#     def forward(self,x,h,c):
#         h_n,c_n = self.lstm(x,(h,c))
#         h_view = h_n.view(-1,1,64,48)
#         out = self.conv1(h_view)
#         return out,h_n,c_n
# class LSTMCell(nn.Module):
#     def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
#         super(LSTMCell, self).__init__()
#         # assert hidden_channels % 2 == 0
#
#         self.input_channels = input_channels
#         self.hidden_channels = hidden_channels
#         self.bias = bias
#         self.kernel_size = kernel_size
#         self.num_features = 4
#
#         self.padding = int((kernel_size-1)/2)
#
#         self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
#         self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
#         self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
#         self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
#         self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
#         self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
#         self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding,  bias=True)
#         self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
#
#         self.Wci = torch.zeros(1, 17, 64, 48).cuda()
#         self.Wcf = torch.zeros(1, 17, 64, 48).cuda()
#         self.Wco = torch.zeros(1, 17, 64, 48).cuda()
#
#
#         # self.dcov1 = nn.ConvTranspose2d(in_channels=1,
#         # out_channels=17,
#         # kernel_size=4,
#         # stride=2,
#         # padding=1)
#
#     def forward(self, x, h, c):
#         print(())
#         ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
#         print(ci.shape)
#         cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
#         cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
#         co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
#         ch = co * torch.tan(cc)
#         # print(cc.shape)
#         return ch, cc


class lstmModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTMCell(17*64*48,32*24)
        self.dcov1 = nn.ConvTranspose2d(in_channels=1,
        out_channels=17,
        kernel_size=4,
        stride=2,
        padding=1)
        # self.conv1 = nn.Conv2d(in_channels=1,
        # out_channels=17,
        # kernel_size=1,
        # stride=1,
        # padding=0)
    def forward(self,x,h,c):
        h_n,c_n = self.lstm1(x,(h,c))
        h_view = h_n.view(-1,1,32,24)
        out = self.dcov1(h_view)
        return out,h_n,c_n
# 训练
def train(config, train_loader, net1, criterion, optimizer1, epoch, output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    end = time.time()
    net1.train()
    result = []
    feature = []

    # net2.train()
    for i, index_names in enumerate(train_loader):
    # for i, (inputs, targets, target_weights, metas) in enumerate(train_loader):
        data_time.update(time.time() - end)
        # h = torch.randn(1,64*48,dtype = torch.float).cuda()
        # c = torch.randn(1,64*48,dtype = torch.float).cuda()
        # h = torch.randn(1,17,64,48,dtype = torch.float).cuda()
        # c = torch.randn(1,17,64,48,dtype = torch.float).cuda()
        h = torch.randn(1,32*24,dtype = torch.float).cuda()
        c = torch.randn(1,32*24,dtype = torch.float).cuda()
        loss = 0

        for n in range(len(index_names)):
            pname = index_names[n][0]
            image = np.load('feature/train_image/' + pname + '.npy')
            image = torch.from_numpy(image)
            # input = np.load('feature/train/' + pname + '.npy')
            input = np.load('/media/cc/00071A630003D02A/posetrack17/train_p/' + pname + '.npy')

            input = torch.from_numpy(input)
            # print(input.shape)
            target = np.load('feature/train_target/' + pname + '.npy')
            target = torch.from_numpy(target)
            # print(target.shape)
            target_weight = np.load('feature/train_weight/' + pname + '.npy')
            target_weight = torch.from_numpy(target_weight)
            # print(target_weight.shape)
            meta_file = Path('feature/train_label/' + pname + '.json')
            with meta_file.open() as f:
                meta = json.load(f)
                # print('meta:',meta)

        # for n in range(len(inputs)):
        #     input = inputs[n]
        #     target = targets[n]
        #
        #     target_weight = target_weights[n]
        #     meta = metas[n]

            # name = meta['image'][0]
            # pname = name[-10:-4]
            # vname = name[-28:-11]


            out1 = input.view(-1,17*64*48).cuda()
            out2,h,c = net1(out1,h,c)#out2:(batch,17,64,48)
            out2 = out2 + input.cuda()
            # h,c = net1(input,h,c)
            # out2 = input.cuda() + h

            # print(out2.shape)

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss += criterion(out2, target, target_weight)
            # print(loss)

            optimizer1.zero_grad()
            # optimizer2.zero_grad()
            loss.backward(retain_graph=True)#retain_graph=True
            optimizer1.step()
            # optimizer2.step()

            losses.update(loss.item(), input.size(0))

            _, avg_acc, cnt, pred = accuracy(out2.detach().cpu().numpy(),
                                             target.detach().cpu().numpy())
            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if n % 100 == 0:
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed {speed:.1f} samples/s\t' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          epoch, i, len(train_loader), batch_time=batch_time,
                          speed=image.size(0)/batch_time.val,
                          data_time=data_time, loss=losses, acc=acc)
                logger.info(msg)

                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer.add_scalar('train_acc', acc.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

                prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
                save_debug_images(config, image, meta, target, pred*4, out2,
                                  prefix)


def validate(config, val_loader, val_dataset, net1, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    net1.eval()
    # net2.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros((21348, 17, 3),
                         dtype=np.float32)
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, index_names in enumerate(val_loader):
        # for i, (inputs, targets, target_weights, metas) in enumerate(val_loader):
            # h = torch.randn(1,64*48,dtype = torch.float).cuda()
            # c = torch.randn(1,64*48,dtype = torch.float).cuda()
            h_0 = torch.randn(1,32*24,dtype = torch.float).cuda()
            c_0 = torch.randn(1,32*24,dtype = torch.float).cuda()
            # h_0 = torch.randn(1,17,64,48,dtype = torch.float).cuda()
            # c_0 = torch.randn(1,17,64,48,dtype = torch.float).cuda()
            # print(h.shape)
            loss = 0

            for n in range(len(index_names)):

                pname = index_names[n][0]
                image = np.load('feature/val_image/' + pname + '.npy')
                image = torch.from_numpy(image)
                # input1 = np.load('feature/val/' + pname + '.npy')
                input1 = np.load('/media/cc/00071A630003D02A/posetrack17/val_p/' + pname + '.npy')
                input1 = torch.from_numpy(input1).cuda()

                target = np.load('feature/val_target/' + pname + '.npy')
                target = torch.from_numpy(target)

                target_weight = np.load('feature/val_weight/' + pname + '.npy')
                target_weight = torch.from_numpy(target_weight)

                meta_file = Path('feature/val_label/' + pname + '.json')
                with meta_file.open() as f:
                    meta = json.load(f)
                    # print(np.array(meta['center'][0]))

                out1 = input1.view(-1,17*64*48).cuda()
                out2,h_n,c_n = net1(out1,h_0,c_0)
                # h_n,c_n = net1(input1,h_0,c_0)
                h_0 = h_n
                c_0 = c_n
                # out2 = h_n
                # print('n:',n)
                # out2 = h_n + input1.cuda()
                out2 = out2 + input1.cuda()
                # print('out:',out2.shape)
                # result_t.append([i,h,c,out2])



                # out = torch.from_numpy(np.load('feature/train/'+ vname + pname + '.npy'))
                # out1 = out.view(-1,17*64*48).cuda()
                # out2,h,c = net1(out1,h,c)#out2:(batch,17,64,48)


                target = target.cuda(non_blocking=True)
                target_weight = target_weight.cuda(non_blocking=True)
                # print('output:',output.shape)

                loss += criterion(out2, target, target_weight)

                num_images = input1.size(0)
                # measure accuracy and record loss
                losses.update(loss.item(), num_images)
                _, avg_acc, cnt, pred = accuracy(out2.cpu().numpy(),
                                                 target.cpu().numpy())
                # print(avg_acc)
                # if avg_acc < 0.1:
                #     logger.info(pname)

                acc.update(avg_acc, cnt)


                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # c = meta['center'].numpy()
                # s = meta['scale'].numpy()
                c = np.array(meta['center'])
                # print(c.shape)
                s = np.array(meta['scale'])
                # score = meta['score'].numpy()

                preds, maxvals = get_final_preds(
                    config, out2.clone().cpu().numpy(), c, s)
                # print(preds.shape)

                all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
                # print('all_preds:',all_preds[idx:idx + num_images, :, 0:2].shape)
                all_preds[idx:idx + num_images, :, 2:3] = maxvals
                # double check this all_boxes parts
                all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
                all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
                all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
                # all_boxes[idx:idx + num_images, 5] = score
                # image_path.extend(meta['image'])
                # if config.DATASET.DATASET == 'posetrack':
                #     filenames.extend(meta['filename'])
                #     imgnums.extend(meta['imgnum'].numpy())

                idx += num_images
                # print('idx:',idx)
                # print('all_pred:',len(all_preds))

                if n % 5 == 0:
                    msg = 'Test: [{0}/{1}]\t' \
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                          'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                              i, len(val_loader), batch_time=batch_time,
                              loss=losses, acc=acc)
                    logger.info(msg)

                    prefix = '{}_{}'.format(os.path.join(output_dir, 'val'), i)
                    save_debug_images(config, image, meta, target, pred*4, out2,
                                      prefix)




        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums)


        _, full_arch_name = get_model_name(config)
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, full_arch_name)
        else:
            _print_name_value(name_values, full_arch_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', losses.avg, global_steps)
            writer.add_scalar('valid_acc', acc.avg, global_steps)
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars('valid', dict(name_value), global_steps)
            else:
                writer.add_scalars('valid', dict(name_values), global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def main():
    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED





    #特征提取模型加载
    # model_state_file_dir = '/media/cc/00035E420005A4F5/exper/lstmpose/backup/256x192_d256x3_adam_lr1e-3'
    # model_state_file = os.path.join(model_state_file_dir, 'final_state.pth.tar')
    # model = eval('models.'+'singleModel'+'.get_pose_net')(config, is_train=False)
    # model = eval('models.'+'posemodel'+'.get_pose_net')(config, is_train=False)
    #
    # pretrained_dict = torch.load(model_state_file)
    # model_dict = model.state_dict()
    # # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # # model_dict.update(pretrained_dict)
    # model.load_state_dict(pretrained_dict)
    # print(model)


    # model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    # print(model)
    # model.load_state_dict(torch.load(model_state_file))

    #LSTMcell和分类模型构建
    net1 = lstmModel().cuda()
    # net1 = LSTMCell(17,17,1).cuda()
    # print(net1)
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }
    dump_input = torch.rand((config.TRAIN.BATCH_SIZE,
                             3,
                             config.MODEL.IMAGE_SIZE[1],
                             config.MODEL.IMAGE_SIZE[0]))
    # writer_dict['writer'].add_graph(net1, (dump_input, ), verbose=False)
    # gpus = [int(i) for i in config.GPUS.split(',')]
    # net1 = torch.nn.DataParallel(net1, device_ids=gpus).cuda()
    # net1 = net1.cuda()
    # os.system("pause")
    #
    #
    # os.system("pause")


    # net2 = getPose()
    # pretrained_dict1 = torch.load(model_state_file)
    # model_dict1 = net2.state_dict()
    # pretrained_dict1 = {k: v for k, v in pretrained_dict1.items() if k in model_dict1}
    # model_dict1.update(pretrained_dict1)
    # net2.load_state_dict(model_dict1)
    # net2 = torch.nn.DataParallel(net2, device_ids=gpus).cuda()

    # print(net2)
    # print(model_dict1)

    optimizer1=torch.optim.Adam(net1.parameters(),lr=0.001)
    # optimizer2=torch.optim.Adam(net2.parameters(),lr=0.001)
    lr_scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, [90,120], 0.1 )
    # lr_scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, [90,120], 0.1 )
    criterion = JointsMSELoss(
        use_target_weight=True
    ).cuda()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = eval('dataset.newdata.PoseTrackDataset')(
        '/media/cc/00035E420005A4F5/exper/lstmpose/feature',
        'train_index',
        True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_dataset = eval('dataset.newdata.PoseTrackDataset')(
        '/media/cc/00035E420005A4F5/exper/lstmpose/feature',
        'val_index',
        False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )



    best_perf = 0.0
    best_model = False
    for epoch in range(10):
        lr_scheduler1.step()
        # lr_scheduler2.step()


        # train for one epoch
        train(config, train_loader, net1, criterion, optimizer1, epoch,
              final_output_dir, tb_log_dir, writer_dict)

        perf_indicator = validate(config, valid_loader, valid_dataset, net1,
                                  criterion, final_output_dir, tb_log_dir,
                                  writer_dict)

        if perf_indicator > best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': get_model_name(config),
            'state_dict': net1.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer1.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(net1.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
