##############################################################################
####                             Yurong Chen                              ####
####                      chenyurong1998 AT outlook.com                   ####
####                           Hunan University                           ####
####                       Happy Coding Happy Ending                      ####
##############################################################################

import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import cv2
import torch
import argparse
import numpy as np
from func import *
import scipy.io as sio
import matplotlib.pyplot as plt
from optimization import ADMM_Iter
import warnings

warnings.filterwarnings('ignore')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
np.random.seed(1234)

#-----------------------Opti. Configuration -----------------------#
parser = argparse.ArgumentParser()
parser.add_argument('--iter_num', default = 40,             help="Maximum number of iterations")
parser.add_argument('--lambda_',  default = 0.03,           help="Facotr of the LCTC regularization")
parser.add_argument('--LR_iter',  default = 1200,           help="Training epochs of CTC networks")
parser.add_argument('--R_iter',   default = 1000,           help="Reduced Training epochs of CTC networks")
parser.add_argument('--lambda_R', default = 0.1,            help="Factor of TV/SSTV regularization in CTC")
parser.add_argument('--ip_BI',    default = 8,              help="The number of channel of input")
parser.add_argument('--case',     default = 'Case6',        help="Case1-6")
args = parser.parse_args()


#----------------------- Data Configuration -----------------------#
dataset_dir = '../Data/CAVE_Noisy_data/'
data_name = args.case
if args.iter_num == 1:
    args.LR_iter = 5200
    args.lambda_R = args.lambda_R*10
    print('---------------- Adjust lambda_R ----------------', data_name)

results_dir = './Results/' + data_name + '/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
matfile = dataset_dir + '/' + data_name + '.mat'

data_truth = torch.from_numpy(sio.loadmat(matfile)['img'])
noisy_data = torch.from_numpy(sio.loadmat(matfile)['noisy_img'])
noisy_data[noisy_data > 1] = 1
noisy_data[noisy_data < 0] = 0
print('Noise Image PSNR:', calculate_psnr(data_truth, noisy_data), 'Noise Image SSIM:', calculate_ssim(data_truth, noisy_data))


#-------------------------- Optimization --------------------------#
x_rec = ADMM_Iter(noisy_data.to(device), data_truth.to(device), args, index = int(data_name[-1:]), save_path = results_dir)
if os.path.exists('./Results/model_weights.pth'):
    os.remove('./Results/model_weights.pth')