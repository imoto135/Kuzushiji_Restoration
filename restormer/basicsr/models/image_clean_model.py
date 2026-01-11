import importlib
import torch
import torch.nn as nn
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
import numpy as np
import cv2

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

import os
import random
import torch.nn.functional as F
from functools import partial

# IoU計算関数
def calculate_iou(img1, img2, threshold=10, **kwargs):
    if len(img1.shape) == 3: img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3: img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    _, mask1 = cv2.threshold(img1, threshold, 255, cv2.THRESH_BINARY)
    _, mask2 = cv2.threshold(img2, threshold, 255, cv2.THRESH_BINARY)
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
    return iou_score
setattr(metric_module, 'calculate_iou', calculate_iou)


class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device
        self.use_identity = use_identity
        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1,1)).item()
        r_index = torch.randperm(target.size(0)).to(self.device)
        target = lam * target + (1-lam) * target[r_index, :]
        input_ = lam * input_ + (1-lam) * input_[r_index, :]
        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments)-1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_

class ImageCleanModel(BaseModel):
    def __init__(self, opt):
        super(ImageCleanModel, self).__init__(opt)
        self.mixing_flag = self.opt['train'].get('mixing_augs', {}).get('mixup', False)
        if self.mixing_flag:
            mixup_beta = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
            use_identity = self.opt['train']['mixing_augs'].get('use_identity', False)
            self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)
        
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            logger = get_root_logger()
            logger.info(f'Loading model from {load_path} for fine-tuning.')
            
            loaded_state_dict = torch.load(load_path, map_location=torch.device('cpu'))
            if 'params' in loaded_state_dict:
                loaded_state_dict = loaded_state_dict['params']

            new_state_dict = OrderedDict()
            for k, v in loaded_state_dict.items():
                new_key = f'restormer.{k}'
                new_state_dict[new_key] = v
            
            self.net_g.load_state_dict(new_state_dict, strict=False) 
            logger.info("Fine-tuning: Loaded weights for the 'restormer' part.")

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = define_network(self.opt['network_g']).to(self.device)
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)
            self.net_g_ema.eval()

        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(self.device)
            self.pixel_loss_weight = self.opt['train'].get('pixel_opt', {}).get('loss_weight', 1.0)
        else: self.cri_pix = None

        if train_opt.get('class_opt'):
            class_type = train_opt['class_opt'].pop('type')
            if class_type == 'CrossEntropyLoss': self.cri_class = nn.CrossEntropyLoss()
            else: self.cri_class = getattr(loss_module, class_type)(**train_opt['class_opt']).to(self.device)
            self.class_loss_weight = self.opt['train'].get('class_opt', {}).get('loss_weight', 1.0)
        else: self.cri_class = None
        
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supported yet.')
        self.optimizers.append(self.optimizer_g)

    # --- ★★★ ここからが修正箇所 ★★★ ---
    def feed_train_data(self, data):
        """学習用のデータをモデルに供給するメソッド"""
        self.lq = data['lq'].to(self.device)
        if 'gt' in data: self.gt = data['gt'].to(self.device)
        if 'label' in data: self.label = data['label'].to(self.device)
        
        # mixupなどの拡張は学習時にのみ適用
        if self.mixing_flag:
             self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):
        """検証・テスト用のデータをモデルに供給するメソッド"""
        self.lq = data['lq'].to(self.device)
        if 'gt' in data: self.gt = data['gt'].to(self.device)
        if 'label' in data: self.label = data['label'].to(self.device)
    # --- ここまで修正 ---

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        restored_image, class_logits = self.net_g(self.lq)
        self.output = restored_image
        loss_dict = OrderedDict()
        l_total = 0
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += self.pixel_loss_weight * l_pix
            loss_dict['l_pix'] = l_pix
        if self.cri_class and class_logits is not None:
            l_class = self.cri_class(class_logits, self.label)
            l_total += self.class_loss_weight * l_class
            loss_dict['l_class'] = l_class
        l_total.backward()
        if self.opt['train'].get('use_grad_clip', True):
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        net = self.net_g_ema if hasattr(self, 'net_g_ema') else self.net_g
        net.eval()
        with torch.no_grad():
            output = net(self.lq)
            if isinstance(output, tuple): self.output = output[0]
            else: self.output = output
        if self.opt['is_train']:
            net.train()

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        if dataloader is None:
            get_root_logger().warning("Validation dataloader is None. Skipping validation.")
            return 0.0

        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        
        pbar = tqdm(dataloader, desc=f'Validation on {dataset_name}')
        
        total_images = 0
        
        for idx, val_data in enumerate(pbar):
            total_images += 1
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
            
            del self.lq, self.output
            if 'gt' in visuals: del self.gt
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], f'{current_iter}', f'{img_name}.png')
                else:
                    save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}.png')
                
                lq_img = tensor2img([visuals['lq']], rgb2bgr=rgb2bgr)
                if 'gt' in visuals:
                    h, w, _ = lq_img.shape
                    if gt_img.shape[0] != h or gt_img.shape[1] != w: gt_img = cv2.resize(gt_img, (w, h), interpolation=cv2.INTER_NEAREST)
                    if sr_img.shape[0] != h or sr_img.shape[1] != w: sr_img = cv2.resize(sr_img, (w, h), interpolation=cv2.INTER_NEAREST)
                    combined_img = np.concatenate((gt_img, lq_img, sr_img), axis=1)
                else:
                    combined_img = np.concatenate((lq_img, sr_img), axis=1)
                imwrite(combined_img, save_img_path)

            if with_metrics:
                if self.opt['val'].get('binarize_for_metrics', False):
                    sr_img_gray = cv2.cvtColor(sr_img, cv2.COLOR_BGR2GRAY)
                    _, sr_img_bin = cv2.threshold(sr_img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    sr_img_for_metric = cv2.cvtColor(sr_img_bin, cv2.COLOR_GRAY2BGR)
                else:
                    sr_img_for_metric = sr_img

                for name, opt_ in deepcopy(self.opt['val']['metrics']).items():
                    metric_type = opt_.pop('type')
                    self.metric_results[name] += getattr(metric_module, metric_type)(sr_img_for_metric, gt_img, **opt_)
        
        current_metric = 0.0
        if with_metrics:
            if total_images > 0:
                for metric in self.metric_results.keys():
                    self.metric_results[metric] /= total_images
                current_metric = list(self.metric_results.values())[0]
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        return current_metric

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter, **kwargs):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter, **kwargs)

