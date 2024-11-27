##############################################################################
####                             Yurong Chen                              ####
####                      chenyurong1998 AT outlook.com                   ####
####                           Hunan University                           ####
####                       Happy Coding Happy Ending                      ####
##############################################################################

import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import torch
import numpy as np
from func import *
from numpy import *
import scipy.io as sio
from torchmetrics import SpectralAngleMapper, ErrorRelativeGlobalDimensionlessSynthesis
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main(data_name):
    #----------------------- Data Configuration -----------------------#
    print('\n')
    print('Test Scene:', data_name)
    
    dataset_dir = '../Data/KAIST_Dataset/Orig_data/'
    matfile = dataset_dir + data_name + '.mat'
    data_truth = torch.from_numpy(sio.loadmat(matfile)['img'])
    recon = torch.from_numpy(sio.loadmat('./Results/' + data_name + './' + data_name + '.mat')['img'])

    sam = SpectralAngleMapper()
    vrecon = recon.double().cpu()
    ssim_ = calculate_ssim(data_truth, vrecon)
    psnr_ = calculate_psnr(data_truth, vrecon)
    sam_ = sam(torch.unsqueeze(vrecon.permute(2, 0, 1), 0).double(), torch.unsqueeze(data_truth.permute(2, 0, 1), 0).double())
    print('PSNR {:2.3f}, ---------, SSIM {:2.3f}, ---------, SAM {:2.3f}'.format(psnr_, ssim_, sam_))


    #x = vrecon.clamp_(0, 1).numpy()
    #data_truth = data_truth.numpy()
    #psnr_ = [cpsnr(x[..., kv], data_truth[..., kv]) for kv in range(28)]
    #ssim_ = [cssim(x[..., kv], data_truth[..., kv]) for kv in range(28)]
    #print('---------- PNSR:', np.mean(psnr_), '---------- SSIM:', np.mean(ssim_))
    


data_list = ['scene01', 'scene02', 'scene03', 'scene04', 'scene05', 'scene06', 'scene07', 'scene08', 'scene09', 'scene10']
for file_name in data_list:
    main(file_name)
