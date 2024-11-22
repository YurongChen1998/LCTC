##############################################################################
####                             Yurong Chen                              ####
####                      chenyurong1998 AT outlook.com                   ####
####                           Hunan University                           ####
####                       Happy Coding Happy Ending                      ####
##############################################################################

import torch
import time
import scipy.io as sio
from func import *
from thop import profile
from model.model_loader import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')




def ADMM_Iter(meas, Phi, truth_tensor, args):
    #-------------- Initialization --------------#
    x0 = shift_back(At(meas, Phi), 2)
    z, u = x0.to(device), torch.zeros_like(x0).to(device)
    Phi_sum = torch.sum(Phi, 2)
    Phi_sum[Phi_sum==0] = 1
    im_input = get_input([1, args.ip_BI, x0.shape[0], x0.shape[1]]).to(device)
    best_PSNR, iter_num, lambda_ = 0, args.iter_num, args.lambda_

    # ---------------- Iteration ----------------#
    for iter in range(iter_num):
        x = z.to(device) - u.to(device)
        x = x + shift_back(At((meas - A(shift(x, 2), Phi))/(Phi_sum + lambda_), Phi), 2)

        # --------------- Evaluation ---------------#
        psnr_x = calculate_psnr_tensor(truth_tensor, x.permute(2, 0, 1).squeeze(0))
        print('Iter {} | PSNR = {:.2f}dB'.format( iter, psnr_x))

        z = x+u
        z = PnP_MoLi(meas, Phi, z.permute(2, 0, 1).unsqueeze(0), truth_tensor, im_input, args)
        u = u + (x.to(device) - z.to(device))
    return z




def PnP_MoLi(meas, Phi, z, truth_tensor, im_input, args):
    torch.backends.cudnn.benchmark = True
    iter_num = args.LR_iter
    _, B, _, _ = truth_tensor.shape
    best_loss = float('inf')
    loss_l1 = torch.nn.L1Loss().to(device)
    loss_l2 = torch.nn.MSELoss().to(device)
    im_net = CTC_model_load(args.ip_BI, B)
    
    save_model_weight = False if args.iter_num == 1 else True
    if os.path.exists('Results/model_weights.pth'):
        im_net[0].load_state_dict(torch.load('Results/model_weights.pth'))
        print('----------------------- Load model weights -----------------------')
        iter_num, save_model_weight = args.R_iter, True
        
        
    im_net[0].train()
    input_params = [im_input]
    im_input_temp = im_input.clone()
    net_params = list(im_net[0].parameters())
    params = net_params + input_params
    optimizer = torch.optim.Adam([{'params': params, 'lr': 1e-3}])
    
    #flops, model_size = profile(im_net[0], inputs = (im_input, ))
    #print('------- FLOPs: {:.3f} G'.format(flops/1000**3), '------- Model Size: {:.3f} MB'.format(model_size/1024**2))
    
    begin_time = time.time()
    for idx in range(iter_num):
        im_input_perturbed = im_input + im_input_temp.normal_()*0.033
        model_out = im_net[0](im_input_perturbed)
        pred_meas = A(shift(model_out.squeeze(0).permute(1, 2, 0), 2).to(device), Phi.to(device))

        if z == None:
            loss = args.lambda_R*loss_l1(meas, pred_meas) #+ (1/2)*loss_l2(meas, pred_meas)
        else:
            loss = args.lambda_R*loss_l1(meas, pred_meas) #+ (args.lambda_/2)*loss_l2(z, model_out)

        loss_tv = loss_l1(im_input, torch.zeros_like(im_input))
        loss += args.lambda_*loss_tv
        
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
       
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_hs_recon = model_out.detach()
            if save_model_weight == True:
                torch.save(im_net[0].state_dict(), 'Results/model_weights.pth')
        
        if (idx+1)%200==0:
            PSNR = calculate_psnr_tensor(truth_tensor, model_out.squeeze(0))
            print('Iter {}, x_loss:{:.3f}, s_loss:{:.3f}, PSNR:{:.2f}'.format(idx+1, loss.item(), loss_tv.item(), PSNR))

    end_time = time.time()
    #print('-------------- Finished----------, running time {:.1f} seconds.'.format(end_time - begin_time))
    return best_hs_recon.squeeze(0).permute(1, 2, 0)

