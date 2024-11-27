##############################################################################
####                             Yurong Chen                              ####
####                      chenyurong1998 AT outlook.com                   ####
####                           Hunan University                           ####
####                       Happy Coding Happy Ending                      ####
##############################################################################

import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import torch
import argparse
import numpy as np
from func import *
from numpy import *
import scipy.io as sio
import matplotlib.pyplot as plt
import torch.nn.functional as F
from optimization import ADMM_Iter
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
random.seed(5)



#-----------------------Opti. Configuration -----------------------#
parser = argparse.ArgumentParser()
parser.add_argument('--iter_num', default = 1,              help="Maximum number of iterations")
parser.add_argument('--lambda_',  default = 1,              help="Facotr of the LCTC regularization")
parser.add_argument('--LR_iter',  default = 6000,           help="Training epochs of CTC networks")
parser.add_argument('--R_iter',   default = 850,            help="Reduced Training epochs of CTC networks")
parser.add_argument('--lambda_R', default = 0.07,           help="Factor of TV/SSTV regularization in CTC")
parser.add_argument('--ip_BI',    default = 4,              help="The number of channel of input")
parser.add_argument('--step',     default = 2,              help="step for spectral shifting")
parser.add_argument('--scene',    default = 'scene10',      help="scene01-10")
args = parser.parse_args()


#----------------------- Data Configuration -----------------------#
dataset_dir = '../Data/KAIST_Dataset/Orig_data/'
data_name = args.scene

results_dir = './Results/' + data_name + '/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
matfile = dataset_dir + '/' + data_name + '.mat'

data_truth = torch.from_numpy(sio.loadmat(matfile)['img'])
truth_tensor = data_truth.permute(2, 0, 1).unsqueeze(0).to(device)
_, nC, h, w= truth_tensor.shape
data_truth_shift = torch.zeros((h, w + args.step*(nC - 1), nC))
for i in range(nC):
    data_truth_shift[:, i*args.step:i*args.step+h, i] = data_truth[:, :, i]


#----------------------- Mask Configuration -----------------------#
mask = torch.zeros((h, w + args.step*(nC - 1)))
mask_3d = torch.unsqueeze(mask, 2).repeat(1, 1, nC)
mask_256 = torch.from_numpy(sio.loadmat('../Data/KAIST_Dataset/mask/mask256.mat')['mask'])
for i in range(nC):
    mask_3d[:, i*args.step:i*args.step+h, i] = mask_256
Phi = mask_3d
meas = torch.sum(Phi * data_truth_shift, 2)

'''
Sparse_noise = np.random.choice((0, 1, 2), size=(meas.shape[0], meas.shape[1]), p=[0.97, 0.03/2., 0.03/2.])
#Gauss_noise = np.random.normal(loc=0.5, scale=0.5, size=(meas.shape[0], meas.shape[1]))
#meas = meas + 0.1*Gauss_noise
meas[Sparse_noise == 1] = torch.max(meas)
meas[Sparse_noise == 2] = 0
meas = meas.float()
'''

plt.figure()
plt.imshow(meas, cmap='gray')
plt.savefig(results_dir+'/meas.png')


#-------------------------- Optimization --------------------------#
x_rec = ADMM_Iter(meas.to(device), Phi.to(device), truth_tensor, args)
sio.savemat(results_dir+'/{}.mat'.format(data_name), {'img': x_rec.cpu().numpy()})
if os.path.exists('./Results/model_weights.pth'):
    os.remove('./Results/model_weights.pth')
