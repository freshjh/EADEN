from utils.config import cfg  # isort: split

import os
import time

from tensorboardX import SummaryWriter
from tqdm import tqdm

from utils.datasets import create_dataloader
from utils.earlystop import EarlyStopping
from utils.eval import get_val_cfg, validate
from utils.trainer import Trainer
from utils.utils import Logger
import torch, random, os, numpy as np

# 忽略警告
import warnings
warnings.filterwarnings("ignore")

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False


if __name__ == "__main__":
    val_cfg = get_val_cfg(cfg, split="val", copy=True)
    cfg.dataset_root = os.path.join(cfg.dataset_root, "train")
    
    # 设置随机种子
    seed_torch(cfg.seed)
   
    data_loader = create_dataloader(cfg)
    dataset_size = len(data_loader)


    log = Logger()
    log.open(cfg.logs_path, mode="a")
    log.write("Num of training images = %d\n" % (dataset_size * cfg.batch_size))
    log.write("Config:\n" + str(cfg.to_dict()) + "\n")

    train_writer = SummaryWriter(os.path.join(cfg.exp_dir, "train"))
    val_writer = SummaryWriter(os.path.join(cfg.exp_dir, "val"))

    trainer = Trainer(cfg)
    early_stopping = EarlyStopping(patience=cfg.earlystop_epoch, delta=-0.001, verbose=True)
    for epoch in range(cfg.nepoch):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for data in tqdm(data_loader, dynamic_ncols=True):
            trainer.total_steps += 1
            epoch_iter += cfg.batch_size

            trainer.set_input(data)
            trainer.optimize_parameters()


            train_writer.add_scalar("loss", trainer.loss, trainer.total_steps)


        
        if epoch % cfg.save_epoch_freq == 0:
            log.write("saving the model at the end of epoch %d, iters %d\n" % (epoch, trainer.total_steps))
            trainer.save_networks("latest")
            trainer.save_networks(epoch)


        if cfg.warmup:

            trainer.scheduler.step()


        
        trainer.train()
