import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import os
from torchvision import transforms
from dataset.transform import Resize, NormalizeImage, PrepareForNet
from util.utils import parse_dataset_txt, read_calib_xml
import numpy as np
import random

class Booster(Dataset):
    def __init__(self, filelist_path, mode, size=(518, 518),baseline_factor=1000):
        self.brightness = (0.8, 1.2)
        self.contrast = (0.8, 1.2)
        self.saturation = (0.8, 1.2)
        self.hue = (-0.1, 0.1)
        self.baseline_factor = baseline_factor
        self.root_path = "/data_nvme/jing/NTIRE/train/"
        dataset_dict = parse_dataset_txt(filelist_path)
        self.gt_fileslist = [os.path.join(self.root_path, f) for f in dataset_dict["gt_paths"]]
        self.image_fileslist = [os.path.join(self.root_path, f) for f in dataset_dict["image_paths"]]
        self.max_depth = 10000
        self.min_depth = 1
        if "calib_paths" in dataset_dict:
            self.focalslist = []
            self.baselineslist = []
            for calib_path in dataset_dict["calib_paths"]:
                fx, baseline = read_calib_xml(os.path.join(self.root_path, calib_path))
                self.focalslist.append(fx)
                self.baselineslist.append(baseline)


        self.mode = mode
        self.size = size

        # with open(filelist_path, 'r') as f:
        #     self.filelist = f.read().splitlines()
        
        net_w, net_h = size
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True if mode == 'train' else False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            # ColorAug(prob=0.5),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
    
    def __getitem__(self, item):
        np.seterr(divide='ignore')
        img_path = self.image_fileslist[item]
        gt_path = self.gt_fileslist[item]
        
        # load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        # load gt
        gt = np.load(gt_path).astype(np.float32)
        fx = self.focalslist[item]
        baseline = self.baselineslist[item]
        baseline = baseline * self.baseline_factor

        # CLIP DEPTH GT
        gt[gt > fx * baseline / self.min_depth] = 0 # INVALID IF LESS THAN 1mm (very high disparity values)
        gt[gt < fx * baseline / self.max_depth] = 0 # INVALID IF MORE THAN max_depth meters (very small disparity values)

        # if eval in depth, this is forward
        depth = baseline * fx / gt
        depth[np.isinf(depth)] = 0

        sample = self.transform({'image': image, 'depth': depth})
        sample['image'] = torch.from_numpy(sample['image'])
        # color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        # if random.random() > 0.5:
        #     sample['image'] = color_aug(sample['image'])

        sample['depth'] = torch.from_numpy(sample['depth'])
        sample['valid_mask'] = sample['depth'] > 0
        sample['image_path'] = img_path
        
        return sample

    def __len__(self):
        return len(self.image_fileslist)