import argparse
import logging
import os
import pprint
import random
import matplotlib.pyplot as plt

import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from dataset.hypersim import Hypersim
from dataset.booster import Booster
# from dataset.vkitti2 import VKITTI2
from depth_anything_v2.dpt import DepthAnythingV2
from util.dist_helper import setup_distributed
from util.loss import SiLogLoss,ScaleAndShiftInvariantLoss,compute_scale_and_shift
from util.metric import eval_depth
from util.utils import init_log

os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'OFF'
parser = argparse.ArgumentParser(description='Depth Anything V2 for Metric Depth Estimation')

parser.add_argument('--encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--dataset', default='hypersim', choices=['hypersim', 'booster'])
parser.add_argument('--img-size', default=742, type=int)
parser.add_argument('--min-depth', default=0.001, type=float)
parser.add_argument('--max-depth', default=20, type=float)
parser.add_argument('--epochs', default=120, type=int)
parser.add_argument('--bs', default=2, type=int)
parser.add_argument('--lr', default=0.0000003, type=float)
parser.add_argument('--pretrained-from', type=str)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def main():
    args = parser.parse_args()
    
    warnings.simplefilter('ignore', np.RankWarning)
    
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    #设置分布式训练环境
    rank, world_size = setup_distributed(port=args.port)
    # now = datetime.now()
    # folder_path = os.path.join(args.save_path, now.strftime("%Y-%m-%d_%H-%M-%S"))  

#主进程
    if rank == 0:
        all_args = {**vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        writer = SummaryWriter(args.save_path)
    
    cudnn.enabled = True
    cudnn.benchmark = True
    
    size = (args.img_size, args.img_size)
    if args.dataset == 'hypersim':
        trainset = Hypersim('dataset/splits/hypersim/train.txt', 'train', size=size)
    elif args.dataset == 'booster':
        trainset = Booster('dataset/splits/booster/train.txt', 'train', size=size)
    else:
        raise NotImplementedError
        #分布式采样器
    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=args.bs, pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler)
    
    if args.dataset == 'hypersim':
        valset = Hypersim('dataset/splits/hypersim/val.txt', 'val', size=size)
    elif args.dataset == 'booster':
        valset = Booster('dataset/splits/booster/val.txt', 'val', size=size)
    else:
        raise NotImplementedError
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4, drop_last=True, sampler=valsampler)
    
    local_rank = int(os.environ["LOCAL_RANK"])
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DepthAnythingV2(**{**model_configs[args.encoder]})
    
    if args.pretrained_from:
        model.load_state_dict({k: v for k, v in torch.load(args.pretrained_from, map_location='cpu').items()})
    
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=True)
    
    criterion = ScaleAndShiftInvariantLoss().cuda(local_rank)
   #beta1 一阶矩估计和二阶矩估计的指数衰减率。
   #weight_decay 是权重衰减（也称为 L2 正则化）系数
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    
    total_iters = args.epochs * len(trainloader)
    
    previous_best = {'delta1.25': 0, 'delta1.20': 0, 'delta1.15': 0, 'delta1.10': 0, 'delta1.05': 0, 'mae': 1000, 'absrel': 1000, 'rmse':1000}
    
    for epoch in range(args.epochs):
        if rank == 0:
            logger.info('===========> Epoch: {:}/{:}, delta1.25: {:.3f}, delta1.20: {:.3f}, delta1.15: {:.3f}, delta1.10: {:.3f}, delta1.05: {:.3f}'.format(epoch, args.epochs, previous_best['delta1.25'], previous_best['delta1.20'],previous_best['delta1.15'],previous_best['delta1.10'],previous_best['delta1.05']))
            logger.info('===========> Epoch: {:}/{:}, mae: {:.3f}, absrel: {:.3f}, rmse: {:.3f}'.format(
                            epoch, args.epochs, previous_best['mae'], previous_best['absrel'], previous_best['rmse']))
        
        trainloader.sampler.set_epoch(epoch + 1)
        
        model.train()
        total_loss = 0
        
        for i, sample in enumerate(trainloader):
            optimizer.zero_grad()
            
            img, depth, valid_mask = sample['image'].cuda(), sample['depth'].cuda(), sample['valid_mask'].cuda()
            
            if random.random() < 0:
                img = img.flip(-1)
                depth = depth.flip(-1)
                valid_mask = valid_mask.flip(-1)

            pred = model(img)
            loss = criterion(pred, depth, valid_mask)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            iters = epoch * len(trainloader) + i
            #学习率衰减
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            
            optimizer.param_groups[0]["lr"] = lr

            if rank == 0:
                writer.add_scalar('train/loss', loss.item(), iters)
            
            if rank == 0 and i % 5 == 0:
                logger.info('Iter: {}/{}, LR: {:.7f}, Loss: {:.3f}'.format(i, len(trainloader), optimizer.param_groups[0]['lr'], loss.item()))
        
        model.eval()

        results = {'delta1.25': torch.tensor([0.0]).cuda(), 'delta1.20': torch.tensor([0.0]).cuda(), 'delta1.15': torch.tensor([0.0]).cuda(), 
                   'delta1.10': torch.tensor([0.0]).cuda(), 'delta1.05': torch.tensor([0.0]).cuda(), 'mae': torch.tensor([0.0]).cuda(), 
                   'absrel': torch.tensor([0.0]).cuda(), 'rmse': torch.tensor([0.0]).cuda()}
        nsamples = torch.tensor([0.0]).cuda()
        
        for i, sample in enumerate(valloader):
            
            img, depth, valid_mask = sample['image'].cuda().float(), sample['depth'].cuda()[0], sample['valid_mask'].cuda()[0]
            
            with torch.no_grad():
                pred = model(img)
                pred = F.interpolate(pred[:, None], depth.shape[-2:], mode='bilinear', align_corners=True)[0, 0]
            if valid_mask.sum() < 10:
                continue
            scale, shift = compute_scale_and_shift(torch.unsqueeze(pred, axis=0), 
                                                torch.unsqueeze(depth, axis=0), 
                                                torch.unsqueeze((depth > 0).float(), axis=0))
            pred = pred * scale + shift
            cur_results = eval_depth(pred[valid_mask], depth[valid_mask])
            
            for k in results.keys():
                results[k] += cur_results[k]
            nsamples += 1
        #同步进程
        torch.distributed.barrier()
        
        for k in results.keys():
            dist.reduce(results[k], dst=0)
        dist.reduce(nsamples, dst=0)
        
        if rank == 0:
            logger.info('==========================================================================================')
            logger.info('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(results.keys())))
            logger.info('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(*tuple([(v / nsamples).item() for v in results.values()])))
            logger.info('==========================================================================================')
            print()
            
            for name, metric in results.items():
                writer.add_scalar(f'eval/{name}', (metric / nsamples).item(), epoch)
        
        for k in results.keys():
            if k in ['delta1.05','delta1.10','delta1.15','delta1.20','delta1.25']:
                previous_best[k] = max(previous_best[k], (results[k] / nsamples).item())
            else:
                previous_best[k] = min(previous_best[k], (results[k] / nsamples).item())
        
        if epoch % 5==0 and rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, f'epoch_{epoch}.pth'))


if __name__ == '__main__':
    main()
