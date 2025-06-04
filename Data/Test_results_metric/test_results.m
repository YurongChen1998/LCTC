clc; 
%% Test KAIST dataset
data_name = 'scene10';
load(['../../LCTC_4SDCASSI/Results/', data_name, '/', data_name, '.mat']);
x_rec = img;
load(['../KAIST_Dataset/Orig_data/', data_name, '.mat']);
data_truth = img;

addpath('assess_fold')
[PSNR,SSIM,SAM,MQresult] = evaluate(data_truth, x_rec, 256, 256);
PSNR = mean(PSNR);
SSIM = mean(SSIM);
SAM = mean(SAM);
fprintf('-------- PSNR = %2.2f, SSIM = %2.3f SAM = %2.3f \n', PSNR, SSIM, SAM);
clear  all;