# -*- coding: utf-8 -*-
import os
import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from tqdm import tqdm
from options.test_options import TestOptions
import csv
import numpy as np
import cv2
import math
import torch
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    setup_seed(20)
    opt = TrainOptions().parse()
    opt2 = TestOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    data_loader_test = CreateDataLoader(opt2)
    dataset_test = data_loader_test.load_data()

    dataset_size = len(data_loader)
    dataset_size_test = len(data_loader_test)
    print('#training images = %d' % dataset_size)
    print('#test images = %d' % dataset_size_test)
    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_steps = 0
    best_psnr = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        print(opt.epoch_count, opt.niter + opt.niter_decay + 1)

        ssim_test_sum = 0
        psnr_test_sum = 0
        img_test_num = 0
        start = time.time()
        if epoch % 1 == 0:
            with tqdm(total=len(dataset_test) / opt2.batchSize, ascii=True) as tt2:
                tt2.set_description('cal Test PSNR')
                epoch_start_time = time.time()
                iter_data_time = time.time()
                epoch_iter = 0
                for i, data in enumerate(dataset_test):
                    img_test_num = img_test_num + opt2.batchSize
                    model.set_input(data)
                    model.test()
                    tt2.update(1)

                print(img_test_num)
                end = time.time()
                print(end - start)





