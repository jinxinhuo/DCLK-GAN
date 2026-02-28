import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util.visualizer import save_images2
from util import html
from tqdm import tqdm
import numpy as np
import time
import cv2
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import re

def check_img_data_range(img):
    if img.dtype == np.uint8:
        return 255
    else:
        return 1.0

def cal_psnr(img1, img2):
    if type(img1) == torch.Tensor:
        img1 = img1.cpu().data.numpy()
    if type(img2) == torch.Tensor:
        img2 = img2.cpu().data.numpy()
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    return peak_signal_noise_ratio(img1, img2, data_range=check_img_data_range(img1))

def cal_ssim(img1, img2):
    return structural_similarity(img1, img2, multichannel = (len(img1.shape) == 3), data_range = check_img_data_range(img1))

#save image caozuo!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
if __name__ == '__main__':
    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    dataset_path = r'G:/ygl/result_img'
    cur_path = os.path.abspath('..')
    test_filepath = 'resext50_vision1'  #
    model_test_result_path = os.path.join(cur_path, test_filepath)

    opt = TestOptions().parse()
    data_loader_test = CreateDataLoader(opt)
    dataset_test = data_loader_test.load_data()
    print('#testing images = %d' % len(dataset_test))

    path = 'G:/ygl/results/LKATGAN/KAIST'

    model = create_model(opt)
    #加载保存参数
    model.setup(opt)

    output_freq = 1
    ssim_test_sum = 0.0
    psnr_test_sum = 0.0
    img_test_num = 0

    with tqdm(total=len(dataset_test), ascii=True) as tt2:
        tt2.set_description('Testing')
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(dataset_test):
            # print(data['A_paths'])
            img_test_num += opt.batchSize
            model.set_input(data)
            model.test()

            tir = model.get_img_nir(data)
            tirs = np.clip(tir.cpu().data.numpy()[0], 0.0, 255.0).transpose([1, 2, 0]).astype(np.uint8)
            fake = model.get_img_gen(data)
            result = np.clip(fake.cpu().data.numpy()[0], 0.0, 255.0).transpose([1, 2, 0]).astype(np.uint8)
            label = model.get_img_label(data)
            labels = np.clip(label.cpu().data.numpy()[0], 0.0, 255.0).transpose([1, 2, 0]).astype(np.uint8)
            ssim_test = cal_ssim(labels, result)
            psnr_test = cal_psnr(labels, result)
            ssim_test_sum = ssim_test_sum + ssim_test
            psnr_test_sum = psnr_test_sum + psnr_test

            tt2.update(1)
            if ((i + 1) % output_freq == 0):
                result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

                Apath = str(data['A_paths'])
                imgpath = ''.join(re.findall(r'\d+', Apath))
                writePath = path + '/' + imgpath + ".png"

                # writePath = path + '/' + str(i) + ".png"
                # print(writePath)
                cv2.imwrite(writePath, result)

                message = '(epoch: %s, iters: %d, test_ssim: %.5f,test_psnr:%.5f) ' % (
                    'testing', img_test_num, ssim_test_sum / img_test_num, psnr_test_sum / img_test_num)
                print(message)

        ssim_test_avg = ssim_test_sum / img_test_num
        psnr_test_avg = psnr_test_sum / img_test_num

        print("TestSSIM:")
        print(ssim_test_avg)
        print("TestPSNR:")
        print(psnr_test_avg)