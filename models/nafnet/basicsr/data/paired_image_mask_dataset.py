# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
from torch.utils import data as data
from torchvision.transforms.functional import normalize
import torch
import numpy as np
import os
import re
import cv2
import random

from basicsr.data.data_util import paired_paths_from_lmdb
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding


class PairedImageMaskDataset(data.Dataset):
    """Paired image dataset with mask for image restoration.

    Read LQ (Low Quality), GT (Ground Truth), and Mask images.
    The mask is concatenated to the LQ image input if provided.

    There are three modes:
    1. 'lmdb': Use lmdb files.
    2. 'meta_info_file': Use meta information file to generate paths.
    3. 'folder': Scan folders to generate paths.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            dataroot_mask (str): Data root path for mask.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation.
            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
            concat_mask (bool): Whether to concatenate mask to LQ input. Default: False.
    """

    def __init__(self, opt):
        super(PairedImageMaskDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.mask_folder = opt['dataroot_mask'] if 'dataroot_mask' in opt else None
        
        # New option to control concatenation
        self.concat_mask = opt['concat_mask'] if 'concat_mask' in opt else False
        
        # Mask dropout probability (for training robustness)
        # When > 0, randomly zero out mask with this probability during training
        self.mask_dropout_prob = opt.get('mask_dropout_prob', 0.0)
        
        # Mask morphology augmentation (dilation/erosion)
        # Helps model become robust to slight mask boundary errors
        self.mask_morph_prob = opt.get('mask_morph_prob', 0.0)
        self.max_kernel_size = opt.get('max_kernel_size', 5)
        
        self.phase = opt.get('phase', 'train')

        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            if self.mask_folder is not None:
                self.io_backend_opt['db_paths'].append(self.mask_folder)
                self.io_backend_opt['client_keys'].append('mask')
                self.paths = paired_paths_from_lmdb(
                    [self.lq_folder, self.gt_folder, self.mask_folder], ['lq', 'gt', 'mask'])
            else:
                self.paths = paired_paths_from_lmdb(
                    [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            # Not fully supported for robust matching yet, assuming folder scan prefered
            pass 
        else:
            # Custom scan to handle filename suffixes in LQ (e.g. _Ghosting) matching pure GT
            self.paths = self._scan_paired_paths()

    def _scan_paired_paths(self):
        paths = []
        # Scan LQ files
        lq_files = []
        print(f"Scanning LQ folder: {self.lq_folder}")
        for root, _, files in os.walk(self.lq_folder):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tif', '.tiff')):
                    lq_files.append(os.path.join(root, file))
        lq_files.sort()
        print(f"Found {len(lq_files)} LQ files.")

        if len(lq_files) > 0:
            print(f"Example LQ file: {lq_files[0]}")

        for lq_path in lq_files:
            img_name = os.path.basename(lq_path)
            basename, ext = os.path.splitext(img_name)
            
            # Find GT basename: match up to _X..._Y...
            # Example: U+3042_..._X1646_Y1617_Ghosting -> U+3042_..._X1646_Y1617
            match = re.search(r'(.+_X\d+_Y\d+)', basename)
            if match:
                gt_basename = match.group(1)
            else:
                # Fallback: assume strict match if pattern not found
                gt_basename = basename
            
            gt_filename = f"{gt_basename}{ext}"
            gt_path = os.path.join(self.gt_folder, gt_filename)
            
            if os.path.exists(gt_path):
                entry = {'lq_path': lq_path, 'gt_path': gt_path}
                
                if self.mask_folder is not None:
                    # Try LQ filename first (for pred_mask), then GT filename (for gt_mask)
                    mask_path_lq = os.path.join(self.mask_folder, img_name)
                    mask_path_gt = os.path.join(self.mask_folder, gt_filename)
                    
                    if os.path.exists(mask_path_lq):
                        entry['mask_path'] = mask_path_lq
                    elif os.path.exists(mask_path_gt):
                        entry['mask_path'] = mask_path_gt
                    else:
                        # Skip if mask is required but missing
                        continue 
                
                paths.append(entry)
            else:
                # Debug print for mismatch (only first few)
                if len(paths) == 0:
                     print(f"GT not found for {lq_path}. Expected at: {gt_path}")
            
        print(f"Total paired paths found: {len(paths)}")
        return paths

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))

        # Load mask if exists
        img_mask = None
        if 'mask_path' in self.paths[index]:
            mask_path = self.paths[index]['mask_path']
            img_bytes = self.file_client.get(mask_path, 'mask')
            try:
                # Load mask as grayscale
                img_mask = imfrombytes(img_bytes, flag='grayscale', float32=True)
                if len(img_mask.shape) == 2:
                    img_mask = np.expand_dims(img_mask, axis=2)
            except:
                raise Exception("mask path {} not working".format(mask_path))

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            
            # padding
            h, w, _ = img_lq.shape
            h_pad = max(0, gt_size - h)
            w_pad = max(0, gt_size - w)
            if h_pad != 0 or w_pad != 0:
                img_lq = cv2.copyMakeBorder(img_lq, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
                img_gt = cv2.copyMakeBorder(img_gt, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
                if img_mask is not None:
                    img_mask = cv2.copyMakeBorder(img_mask, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)

            # random crop
            # Pass mask with gt if available so it gets same crop
            if img_mask is not None:
                img_gts, img_lqs = paired_random_crop([img_gt, img_mask], [img_lq], gt_size, scale, gt_path)
                img_gt, img_mask = img_gts[0], img_gts[1]
                # img_lqs is unwrapped because len=1
                img_lq = img_lqs
            else:
                img_gts, img_lqs = paired_random_crop([img_gt], [img_lq], gt_size, scale, gt_path)
                # Both unwrapped because len=1
                img_gt = img_gts
                img_lq = img_lqs

            # flip, rotation
            if img_mask is not None:
                img_gt, img_lq, img_mask = augment([img_gt, img_lq, img_mask], self.opt['use_flip'], self.opt['use_rot'])
            else:
                img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'], self.opt['use_rot'])

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        if img_mask is not None:
            img_mask = img2tensor([img_mask], bgr2rgb=False, float32=True)[0]

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        # Concatenate LQ and Mask if mask exists AND concat_mask is True
        if img_mask is not None and self.concat_mask:
            # Mask Morphology Augmentation: randomly dilate or erode mask during training
            # This helps the model become robust to slight mask boundary errors
            if self.phase == 'train' and self.mask_morph_prob > 0:
                if random.random() < self.mask_morph_prob:
                    # Convert to numpy for OpenCV operations
                    mask_np = img_mask.numpy().squeeze()  # Remove channel dim for cv2
                    
                    # Random kernel size (odd numbers only for symmetric kernel)
                    kernel_size = random.randint(1, self.max_kernel_size)
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    
                    # Randomly choose dilate or erode
                    if random.random() < 0.5:
                        mask_np = cv2.dilate(mask_np, kernel, iterations=1)
                    else:
                        mask_np = cv2.erode(mask_np, kernel, iterations=1)
                    
                    # Convert back to tensor
                    img_mask = torch.from_numpy(mask_np).unsqueeze(0).float()
            
            # Mask Dropout: randomly zero out mask during training
            # This helps the model learn to work even when mask is imperfect or missing
            if self.phase == 'train' and self.mask_dropout_prob > 0:
                if random.random() < self.mask_dropout_prob:
                    img_mask = torch.zeros_like(img_mask)
            img_lq = torch.cat([img_lq, img_mask], dim=0)

        return_dict = {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }
        if 'mask_path' in self.paths[index]:
            return_dict['mask_path'] = self.paths[index]['mask_path']
            
        return return_dict

    def __len__(self):
        return len(self.paths)
