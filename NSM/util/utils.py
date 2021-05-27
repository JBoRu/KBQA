import logging
import os
import numpy as np
import torch
import random
from torch.utils.tensorboard import SummaryWriter

def create_logger(args):
    log_file = os.path.join(args.checkpoint_dir, args.experiment_name + ".log")
    logger = logging.getLogger()
    log_level = logging.DEBUG if args.log_level == "debug" else logging.INFO
    logger.setLevel(level=log_level)
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # FileHandler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # StreamHandler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info("PARAMETER" + "-" * 10)
    for attr, value in sorted(args.__dict__.items()):
        logger.info("{}={}".format(attr.upper(), value))
    logger.info("---------" + "-" * 10)

    return logger


def get_dict(data_folder, filename):
    filename_true = os.path.join(data_folder, filename)
    word2id = dict()
    with open(filename_true, encoding='utf-8') as f_in:
        for line in f_in:
            word = line.strip()
            word2id[word] = len(word2id)
    return word2id

def init_seed(seed):
    r""" init random seed for random functions in numpy, torch, cuda and cudnn
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def create_tensorboard(args):
    # create a summary writer using the specified folder name.
    name = args.experiment_name
    writer = SummaryWriter(comment=name)
    return writer