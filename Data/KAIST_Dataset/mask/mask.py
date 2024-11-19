import numpy as np
import scipy.io as sio
import torch

def load_mask(path, size=256):
    ## load mask
    data = sio.loadmat(path)
    mask = data['mask']
    mask_3d_temp = np.zeros([256, 256, 28])
    for i in range(28):
        mask_3d_temp[:, :,  i] = np.roll(mask, 2 * i, axis=0)
    sio.savemat('mask_3d_256.mat', {'mask': mask_3d_temp})
    return mask_3d_temp
    

path = './mask256.mat'
a = load_mask(path)
print(a.shape)
