import argparse
import datetime
import logging
import math
import random
import time
import torch
from os import path as osp
from tqdm import tqdm

from basicsr.data import create_dataloader, create_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import create_model
from basicsr.utils import (MessageLogger, check_resume, get_env_info,
                           get_root_logger, get_time_str, init_tb_logger,
                           init_wandb_logger, make_exp_dirs, mkdir_and_rename,
                           set_random_seed)
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import dict2str, parse

import numpy as np
import os

def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
            print('init dist .. ', args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    return opt


def init_loggers(opt):
    log_file = osp.join(opt['path']['log'],
                        f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # initialize wandb logger before tensorboard logger to allow proper sync:
    if (opt['logger'].get('wandb')
            is not None) and (opt['logger']['wandb'].get('project')
                              is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, (
            'should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join('tb_logger', opt['name']))
    return logger, tb_logger


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loader = None, None
    num_iter_per_epoch = 0
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = create_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'],
                                            opt['rank'], dataset_enlarge_ratio)
            train_loader = create_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio /
                (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info(
                'Training statistics:'
                f'\n\tNumber of train images: {len(train_set)}'
                f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                f'\n\tWorld size (gpu number): {opt["world_size"]}'
                f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'])
            logger.info(
                f'Number of val images/folders in {dataset_opt["name"]}: '
                f'{len(val_set)}')
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loader, total_epochs, total_iters, num_iter_per_epoch


def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=True)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # automatic resume ..
    state_folder_path = 'experiments/{}/training_states/'.format(opt['name'])
    import os
    try:
        states = os.listdir(state_folder_path)
    except:
        states = []

    resume_state = None
    if len(states) > 0:
        max_state_file = '{}.state'.format(max([int(x[0:-6]) for x in states]))
        resume_state = os.path.join(state_folder_path, max_state_file)
        opt['path']['resume_state'] = resume_state

    # load resume states if necessary
    if opt['path'].get('resume_state'):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt['path']['resume_state'],
            map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        resume_state = None

    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt[
                'name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join('tb_logger', opt['name']))

    # initialize loggers
    logger, tb_logger = init_loggers(opt)
    
    # Get wandb run object
    wandb_run = None
    try:
        import wandb
        wandb_run = wandb.run
        if wandb_run is not None:
            logger.info(f'Wandb run initialized: {wandb_run.name} (ID: {wandb_run.id})')
            logger.info(f'Wandb run URL: {wandb_run.get_url()}')
    except:
        pass

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loader, total_epochs, total_iters, num_iter_per_epoch = result

    # create model
    if resume_state:  # resume training
        check_resume(opt, resume_state['iter'])
        model = create_model(opt)
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, "
                    f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        model = create_model(opt)
        start_epoch = 0
        current_iter = 0
    
    # Log model complexity to wandb
    if hasattr(model, 'model_complexity') and wandb_run is not None:
        try:
            wandb_run.summary.update(model.model_complexity)
            logger.info(f'Logged model complexity to wandb: {model.model_complexity}')
        except Exception as e:
            logger.warning(f'Failed to log model complexity to wandb: {e}')

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.'
                         "Supported ones are: None, 'cuda', 'cpu'.")

    # training
    logger.info(
        f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()
    
    # Early Stopping 初期化
    early_stop_cfg = opt['train'].get('early_stop', {})
    use_early_stopping = early_stop_cfg.get('enable', False)
    early_stop_patience = early_stop_cfg.get('patience', 10)
    early_stop_metric = early_stop_cfg.get('metric', 'psnr')
    early_stop_mode = early_stop_cfg.get('mode', 'max')

    best_score = None
    no_improve_count = 0


    iters = opt['datasets']['train'].get('iters')
    batch_size = opt['datasets']['train'].get('batch_size_per_gpu')
    mini_batch_sizes = opt['datasets']['train'].get('mini_batch_sizes')
    gt_size = opt['datasets']['train'].get('gt_size')
    mini_gt_sizes = opt['datasets']['train'].get('gt_sizes')

    groups = np.array([sum(iters[0:i + 1]) for i in range(0, len(iters))])

    logger_j = [True] * len(groups)

    scale = opt['scale']

    epoch = start_epoch
    while current_iter <= total_iters:
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        # --- ★★★ プログレスバーの初期化 ★★★ ---
        pbar = tqdm(total=num_iter_per_epoch, desc=f"Epoch {epoch}/{total_epochs}", unit="iter")
        # Resume時の進捗を反映
        pbar.update(current_iter % num_iter_per_epoch)

        while train_data is not None:
            data_time = time.time() - data_time

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(
                current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))

            
            ### ------Progressive learning ---------------------
            j = ((current_iter>groups) !=True).nonzero()[0]
            if len(j) == 0:
                bs_j = len(groups) - 1
            else:
                bs_j = j[0]

            mini_gt_size = mini_gt_sizes[bs_j]
            mini_batch_size = mini_batch_sizes[bs_j]
            
            if logger_j[bs_j]:
                logger.info('\n Updating Patch_Size to {} and Batch_Size to {} \n'.format(mini_gt_size, mini_batch_size*torch.cuda.device_count())) 
                logger_j[bs_j] = False

            lq = train_data['lq']
            gt = train_data['gt']
            label = gt.clone()  # 'gt' (正解画像) を 'label' としても使用する

            if mini_batch_size < batch_size:
                indices = random.sample(range(0, batch_size), k=mini_batch_size)
                lq = lq[indices]
                gt = gt[indices]
                # 2. バッチサイズ変更時に、labelも同じようにスライスする
                label = label[indices]

            if mini_gt_size < gt_size:
                x0 = int((gt_size - mini_gt_size) * random.random())
                y0 = int((gt_size - mini_gt_size) * random.random())
                x1 = x0 + mini_gt_size
                y1 = y0 + mini_gt_size
                lq = lq[:,:,x0:x1,y0:y1]
                gt = gt[:,:,x0*scale:x1*scale,y0*scale:y1*scale]
            
            # 3. model.feed_train_dataに、取り出した'label'を渡す
            model.feed_train_data({'lq': lq, 'gt': gt, 'label': label})
            model.optimize_parameters(current_iter)

            iter_time = time.time() - iter_time
            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_time, 'data_time': data_time})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)
                
                # Log to wandb
                if wandb_run is not None:
                    try:
                        wandb_log = {'epoch': epoch, 'iter': current_iter}
                        wandb_log.update(model.get_current_log())
                        wandb_log.update({'learning_rate': model.get_current_learning_rate()[0]})
                        wandb_log.update({'iter_time': iter_time, 'data_time': data_time})
                        wandb_run.log(wandb_log, step=current_iter)
                        if current_iter % 1000 == 0:
                            logger.info(f'Logged metrics to wandb at iter {current_iter}: {list(wandb_log.keys())}')
                    except Exception as e:
                        logger.warning(f'Failed to log to wandb at iter {current_iter}: {e}')

            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            # validation
            if opt.get('val') is not None and (current_iter %
                                               opt['val']['val_freq'] == 0):
                # --- ★★★ プログレスバーを一時的に閉じる ★★★ ---
                pbar.close()
                rgb2bgr = opt['val'].get('rgb2bgr', True)
                use_image = opt['val'].get('use_image', True)
                model.validation(val_loader, current_iter, tb_logger,
                                 opt['val']['save_img'], rgb2bgr, use_image)
                
                # Log validation metrics to wandb
                if wandb_run is not None:
                    try:
                        val_metrics = {}
                        # Get metric values from model's metric_results
                        if hasattr(model, 'metric_results'):
                            for metric_name, metric_value in model.metric_results.items():
                                val_metrics[f'val/{metric_name}'] = metric_value
                        if val_metrics:
                            wandb_run.log(val_metrics, step=current_iter)
                            logger.info(f'Logged validation metrics to wandb: {val_metrics}')
                    except Exception as e:
                        logger.warning(f'Failed to log validation metrics to wandb: {e}')
                
                if use_early_stopping:
                    # バリデーション結果からモニタリング対象の指標を取得
                    latest_metrics = model.get_validation_metrics()  # ステップ3で実装
                    current_score = latest_metrics.get(early_stop_metric)

                    if current_score is None:
                        logger.warning(f"Early stopping metric '{early_stop_metric}' not found in validation results.")
                    else:
                        # 改善したかを判定
                        is_better = (best_score is None or
                                    (early_stop_mode == 'max' and current_score > best_score) or
                                    (early_stop_mode == 'min' and current_score < best_score))
                        
                        if is_better:
                            best_score = current_score
                            no_improve_count = 0
                            logger.info(f"Validation {early_stop_metric} improved to {current_score:.4f}")
                            
                            # ベストモデル保存（必要なら）
                            model.save(epoch, current_iter, label='best')
                        else:
                            no_improve_count += 1
                            logger.info(f"No improvement in {early_stop_metric}. Count: {no_improve_count}/{early_stop_patience}")

                        # patience を超えたらトレーニング終了
                        if no_improve_count >= early_stop_patience:
                            logger.info(f"Early stopping triggered at iter {current_iter}. Best {early_stop_metric}: {best_score:.4f}")
                            early_stop_flag = True  # ループの外でも確認できるように
                            break  # 現在の epoch ループを中断
                # --- ★★★ 検証後にプログレスバーを再作成 ★★★ ---
                pbar = tqdm(total=num_iter_per_epoch, desc=f"Epoch {epoch}/{total_epochs}", unit="iter")
                pbar.update(current_iter % num_iter_per_epoch)


            data_time = time.time()
            iter_time = time.time()
            train_data = prefetcher.next()

            # --- ★★★ プログレスバーを更新 ★★★ ---
            pbar.update(1)

        # end of iter
        pbar.close()
        epoch += 1

    # end of epoch

    consumed_time = str(
        datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None:
        model.validation(val_loader, current_iter, tb_logger,
                         opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    main()