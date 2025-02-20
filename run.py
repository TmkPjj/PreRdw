import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
import matplotlib.pyplot as plt


from metric_depth.depth_anything_v2.dpt import DepthAnythingV2

# generate depth map for images in the folder using the trained model
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    parser.add_argument('--root_path', type=str,default = '/data_share/jing/jing/pythons/datasets/NTIRE/train/')
    parser.add_argument('--img_path', type=str,default = './assets/booster_train.txt')
    parser.add_argument('--input_size', type=int, default=518)

    parser.add_argument('--vis_depth', type=str,default = './vis_depth/dv2_metric_ft')
    parser.add_argument('--npy_depth', type=str,default = './npy_depth/dv2_metric_ft')

    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    parser.add_argument('--pred_only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--model_load', type=str, default='./checkpoints/depth_anything_v2_metric_hypersim_vitl.pth')

    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    # create model
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    model_dict = torch.load(args.model_load)
    depth_anything.load_state_dict(model_dict)
    # if you fine tune the model using metric_depth training code, you can use the following code to load the model
    # depth_anything.load_state_dict({key[7:]: weight for key, weight in model_dict['model'].items()})

    depth_anything = depth_anything.to(DEVICE).eval()
    
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames_temp = f.read().splitlines()
                filenames = [os.path.join(args.root_path, line) for line in filenames_temp]  
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
        
    cmap = matplotlib.cm.get_cmap('Spectral_r')
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        raw_image = cv2.imread(filename)
        depth = depth_anything.infer_image(raw_image, args.input_size)
        
        output_file = os.path.join(args.npy_depth,filenames_temp[k].rsplit('/', 2)[0])
        if not os.path.exists(output_file):
            os.makedirs(output_file, exist_ok=True)
        np.save(os.path.join(output_file, os.path.splitext(os.path.basename(filenames_temp[k]))[0] + '.npy'), depth)
        
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        
        outdir = os.path.join(args.vis_depth,filenames_temp[k].rsplit('/', 2)[0])
        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
        if args.pred_only:
            cv2.imwrite(os.path.join(outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), depth)
        else:
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth])
            
            cv2.imwrite(os.path.join(outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), combined_result)