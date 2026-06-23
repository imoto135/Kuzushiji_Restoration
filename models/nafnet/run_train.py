import argparse
import datetime
import logging
import math
import random
import time
import torch
from os import path as osp
from tqdm import tqdm  # 追加

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

    parser.add_argument('--input_path', type=str, required=False, help='The path to the input image. For single image inference only.')
    parser.add_argument('--output_path', type=str, required=False, help='The path to the output image. For single image inference only.')

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

    if args.input_path is not None and args.output_path is not None:
        opt['img_path'] = {
            'input_img': args.input_path,
            'output_img': args.output_path
        }

    return opt


def init_loggers(opt):
    log_file = osp.join(opt['path']['log'],
                        f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # initialize wandb logger before tensorboard logger to allow proper sync:
    wandb_run = None
    if (opt['logger'].get('wandb')
            is not None) and (opt['logger']['wandb'].get('project')
                              is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, (
            'should turn on tensorboard when using wandb')
        import wandb
        wandb_run = init_wandb_logger(opt)
        # Log model architecture and hyperparameters
        if wandb_run:
            wandb.config.update({
                'network_type': opt['network_g']['type'],
                'img_channel': opt['network_g'].get('img_channel', 3),
                'out_channel': opt['network_g'].get('out_channel', 3),
                'optimizer': opt['train']['optim_g']['type'],
                'learning_rate': opt['train']['optim_g']['lr'],
                'batch_size': opt['datasets']['train']['batch_size_per_gpu'],
                'total_epochs': opt['train'].get('total_epoch', 'N/A'),
            })
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join('logs', opt['name']))
    return logger, tb_logger, wandb_run


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loader = None, None
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
            
            # エポック数を優先、total_iter は計算値として扱う
            if 'total_epoch' in opt['train']:
                total_epochs = int(opt['train']['total_epoch'])
                total_iters = total_epochs * num_iter_per_epoch
            else:
                total_iters = int(opt['train']['total_iter'])
                total_epochs = math.ceil(total_iters / num_iter_per_epoch)
            
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
        logging.info(f'Resuming from state: {state_folder_path}')
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
    logger, tb_logger, wandb_run = init_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loader, total_epochs, total_iters, num_iter_per_epoch = result

    # create model
    model = create_model(opt)

    # ----- Calculate and log FLOPs and Parameters -----
    # NOTE: Must be done BEFORE torch.compile for accurate measurement
    try:
        from thop import profile, clever_format
        net_g = model.net_g
        net_g.eval()
        
        # Get input dimensions from network config
        img_channel = opt['network_g'].get('img_channel', 3)
        gt_size = opt['datasets']['train'].get('gt_size', 128)
        
        dummy_input = torch.randn(1, img_channel, gt_size, gt_size).to(model.device)
        macs, params = profile(net_g, inputs=(dummy_input,), verbose=False)
        flops = macs * 2  # FLOPs ≈ 2 * MACs
        
        # Format with proper units (e.g., "4.024G", "29.101M")
        formatted = clever_format([macs, params, flops], "%.3f")
        macs_str, params_str, flops_str = formatted[0], formatted[1], formatted[2]
        
        logger.info(f'=' * 60)
        logger.info(f'Model Complexity:')
        logger.info(f'  Input size: {img_channel} x {gt_size} x {gt_size}')
        logger.info(f'  Parameters: {params_str}')
        logger.info(f'  MACs: {macs_str}')
        logger.info(f'  FLOPs: {flops_str}')
        logger.info(f'=' * 60)
        
        # Log to wandb
        if wandb_run:
            import wandb
            wandb.log({
                'model/parameters': params,
                'model/macs': macs,
                'model/flops': flops,
            }, step=0)
            wandb.run.summary['model_params'] = params_str
            wandb.run.summary['model_macs'] = macs_str
            wandb.run.summary['model_flops'] = flops_str
        
        net_g.train()
    except ImportError:
        logger.warning('thop not installed. Skipping FLOPs calculation. Install with: pip install thop')
    except Exception as e:
        logger.warning(f'FLOPs calculation failed: {e}')
    # --------------------------------------------------
    
    # Apply torch.compile for 20-43% speedup (PyTorch 2.x feature)
    if hasattr(torch, 'compile') and opt['train'].get('use_torch_compile', True):
        logger.info('Applying torch.compile to net_g for optimization...')
        model.net_g = torch.compile(model.net_g)

    # resume training
    start_epoch = 0
    current_iter = 0
    if resume_state:
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
        model.resume_training(resume_state)
        logger.info(f"Resuming training from epoch: {start_epoch}, iter: {current_iter}.")


    # ----- early stopping 用の変数 -----
    es_cfg = opt['train'].get('early_stop', None)
    if es_cfg is not None:
        es_metric = es_cfg.get('metric', 'psnr')
        es_mode = es_cfg.get('mode', 'max')
        es_patience = int(es_cfg.get('patience', 10))
        best_score = None
        bad_counts = 0
    # ----------------------------------

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)
    
    # total_iters を MessageLogger に設定（total_epoch 使用時）
    if 'total_epoch' in opt['train']:
        msg_logger.set_max_iters(total_iters)

    # tqdm プログレスバー（イテレーション単位）
    pbar = tqdm(
        total=total_iters, 
        initial=current_iter, 
        ncols=120, 
        unit='iter',
        desc=f'Epoch {start_epoch}/{total_epochs}',
        dynamic_ncols=True
    )

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

    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_time = time.time() - data_time

            current_iter += 1
            if current_iter > total_iters:
                break

            # update learning rate
            model.update_learning_rate(
                current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            # training
            model.feed_data(train_data, is_val=False)
            result_code = model.optimize_parameters(current_iter, tb_logger)

            iter_time = time.time() - iter_time
            
            # Update progress bar
            loss_dict = model.get_current_log()
            loss_str = ' '.join([f'{k}:{v:.3f}' for k, v in loss_dict.items() if 'l_' in k or 'loss' in k.lower()])
            lr = model.get_current_learning_rate()[0]
            pbar.set_description(f'E{epoch}/{total_epochs} | lr:{lr:.1e} | {loss_str}')
            pbar.update(1)
            
            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter, 'total_iter': total_iters}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_time, 'data_time': data_time})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)
                
                # Log to wandb
                if wandb_run:
                    import wandb
                    wandb_log = {
                        'train/epoch': epoch,
                        'train/iter': current_iter,
                        'train/lr': lr,
                    }
                    for k, v in loss_dict.items():
                        wandb_log[f'train/{k}'] = v
                    wandb.log(wandb_log, step=current_iter)

            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            # validation
            if opt['val']['val_freq'] is not None and current_iter % opt['val']['val_freq'] == 0:
                rgb2bgr = opt['val'].get('rgb2bgr', True)
                use_image = opt['val'].get('use_image', True)
                model.validation(
                    dataloader=val_loader,
                    current_iter=current_iter,
                    tb_logger=tb_logger,
                    save_img=opt['val'].get('save_img', False))

                log_dict = model.get_current_log()
                # Format validation metrics manually
                val_message = "Validation metrics: "
                for k, v in log_dict.items():
                    val_message += f"{k}: {v:.4f}  "
                logger.info(val_message)
                
                # Log validation metrics to wandb
                if wandb_run:
                    import wandb
                    val_wandb_log = {'val/iter': current_iter}
                    for k, v in log_dict.items():
                        val_wandb_log[f'val/{k}'] = v
                    wandb.log(val_wandb_log, step=current_iter)

                # ===== early stopping 判定 =====
                if es_cfg is not None:
                    score = log_dict.get(es_metric, None)
                    if score is not None:
                        improved = False
                        if best_score is None:
                            improved = True
                        elif es_mode == 'max' and score > best_score:
                            improved = True
                        elif es_mode == 'min' and score < best_score:
                            improved = True

                        if improved:
                            best_score = score
                            bad_counts = 0
                            logger.info(
                                f'EarlyStop: {es_metric} improved to {score:.4f}')
                            # Log best score to wandb
                            if wandb_run:
                                import wandb
                                wandb.run.summary[f'best_{es_metric}'] = score
                                wandb.run.summary['best_iter'] = current_iter
                        else:
                            bad_counts += 1
                            logger.info(
                                f'EarlyStop: {es_metric} not improved '
                                f'({bad_counts}/{es_patience})')
                            if bad_counts >= es_patience:
                                logger.info(
                                    'Early stopping triggered. Stop training.')
                                pbar.close()
                                return
                # ===============================

            data_time = time.time()
            iter_time = time.time()
            train_data = prefetcher.next()
        # end of iter
        pass  # Progress bar updated per iteration

    pbar.close()
    consumed_time = str(
        datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1: the latest
if __name__ == '__main__':
    main()