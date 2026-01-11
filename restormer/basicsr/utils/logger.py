import datetime
import logging
import time
import torch
from collections import OrderedDict
import random
import string

from basicsr.utils.dist_util import get_dist_info, master_only

def generate_id(length=8):
    """Generates a random ID for wandb."""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))

# --- ★★★ 追加: 欠落していた関数を再追加 ★★★ ---
def get_env_info():
    """環境情報を取得する"""
    import torch
    import torchvision

    msg = '\nVersion Information:'
    msg += f'\n\tPyTorch: {torch.__version__}'
    msg += f'\n\tTorchVision: {torchvision.__version__}'
    return msg
# --- ここまで ---

class MessageLogger():
    """Message logger for printing.

    Args:
        opt (dict): Config. It contains the following keys:
            name (str): Exp name.
            logger (dict): Contains 'print_freq' and 'save_checkpoint_freq'.
            train (dict): Contains 'total_iter'.
        start_iter (int): Start iter.
        tb_logger (obj:`tb_logger`): Tensorboard logger.
    """

    def __init__(self, opt, start_iter=1, tb_logger=None):
        self.exp_name = opt['name']
        self.interval = opt['logger']['print_freq']
        self.start_iter = start_iter
        self.max_iters = opt['train']['total_iter']
        self.tb_logger = tb_logger
        self.start_time = time.time()
        self.logger = get_root_logger()

    def __call__(self, log_vars):
        """Format logging message.

        Args:
            log_vars (dict): It contains the following keys:
                epoch (int): Current epoch.
                iter (int): Current iteration.
                lrs (list): List for learning rates.
                time (float): Iteration time.
                data_time (float): Data time for each iteration.
        """
        # epoch, iter, learning rates
        message = (f'[{self.exp_name[:5]}..][epoch:{log_vars["epoch"]:3d}, '
                   f'iter:{log_vars["iter"]:8,d}, ')
        lrs = log_vars['lrs']
        message += f'lr:('
        for lr in lrs:
            message += f'{lr:.3e},'
        message = message[:-1] + ')] '

        # time and estimated time
        total_time = time.time() - self.start_time
        time_sec_avg = total_time / (log_vars['iter'] - self.start_iter + 1)
        eta_sec = time_sec_avg * (self.max_iters - log_vars['iter'] - 1)
        eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
        message += f'[eta: {eta_str}, '
        message += f'time (data): {log_vars["time"]:.3f} ({log_vars["data_time"]:.3f})] '

        # other items, especially losses
        for k, v in log_vars.items():
            if k not in [
                    'epoch', 'iter', 'lrs', 'time', 'data_time'
            ]:
                message += f'{k}: {v:.4e} '
                # tensorboard logger
                if self.tb_logger is not None and 'debug' not in self.exp_name:
                    self.tb_logger.add_scalar(k, v, log_vars['iter'])
        self.logger.info(message)


@master_only
def init_wandb_logger(opt):
    """Initialize wandb logger.

    Args:
        opt (dict): Config. It contains the following keys:
            logger (dict): Contains 'wandb' dict.
    """
    import wandb
    logger = get_root_logger()

    project = opt['logger']['wandb']['project']
    resume_id = opt['logger']['wandb'].get('resume_id')
    if resume_id:
        wandb_id = resume_id
        resume = 'allow'
        logger.info(f'Resume wandb logger with id={wandb_id}.')
    else:
        # wandb.util.generate_id() は古いので、自作のID生成関数を使う
        wandb_id = generate_id()
        resume = None

    wandb.init(
        id=wandb_id,
        resume=resume,
        name=opt['name'],
        config=opt,
        project=project,
        sync_tensorboard=True)

    logger.info(f"Use wandb logger with id={wandb_id}; project={project}.")


def get_root_logger(logger_name='basicsr',
                    log_level=logging.INFO,
                    log_file=None):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler
    will also be added.

    Args:
        logger_name (str): Logger name.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.
        log_file (str | None): The log filename.

    Returns:
        logging.Logger: The root logger.
    """
    logger = logging.getLogger(logger_name)
    # if the logger has been initialized, just return it
    if logger.hasHandlers():
        return logger

    format_str = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(format=format_str, level=log_level)
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel(logging.ERROR)
    elif log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    return logger


def init_tb_logger(log_dir):
    from torch.utils.tensorboard import SummaryWriter
    tb_logger = SummaryWriter(log_dir=log_dir)
    return tb_logger

