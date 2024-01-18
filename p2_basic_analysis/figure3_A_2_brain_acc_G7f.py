## 
## this code is used for calculate and plot acc map for G7f
## realign should be down in Fiji!
##

import os
from skimage import io
import numpy as np
from utils.projection_3d import *
import cv2

## path
acc_source = [ './p5_representation_analysis/results/Ach-ver16/', './p5_representation_analysis/results/5HT-ver13/']

save_path = '../results/3.odor_random_90_times_OCT_MCH_EA_new/'

data_source = [ '../results/3.odor_random_90_times_OCT_MCH_EA_new/nsyb-G7f-rAch1h/',
               '../results/3.odor_random_90_times_OCT_MCH_EA_new/nsyb-G7f-r5HT1.0/']

flyname_list = [[           
            '20230417-nsyb-G7f-rAch1h/fly2',
           '20230420-nsyb-G7f-rAch1h/fly2',
           '20230420-nsyb-G7f-rAch1h/fly3',
           '20230428-nsyb-G7f-rAch1h/fly1',
           '20230507-nsyb-G7f-rAch1h/fly1',
           '20230510-nsyb-G7f-rAch1h/fly1',
           '20230510-nsyb-G7f-rAch1h/fly2',
           '20230511-nsyb-G7f-rAch1h/fly2',
           '20230511-nsyb-G7f-rAch1h/fly3',
           '20230515-nsyb-G7f-rAch1h/fly1',
           ],
            [
           '20230429-nsyb-G7f-r5HT1.0/fly1',
           '20230506-nsyb-G7f-r5HT1.0/fly1',
           '20230513-nsyb-G7f-r5HT1.0/fly1',
           '20230513-nsyb-G7f-r5HT1.0/fly2',
           '20230516-nsyb-G7f-r5HT1.0/fly2',
           '20230516-nsyb-G7f-r5HT1.0/fly4',
           '20230517-nsyb-G7f-r5HT1.0/fly1',
           '20230601-nsyb-G7f-r5HT1.0/fly1',
           '20230601-nsyb-G7f-r5HT1.0/fly3',
           '20230603-nsyb-G7f-r5HT1.0/fly1',
           ]]


## parameter set
voxel_size = [2,4,4]

pixel_size = [318.51/256*4,318.51/256*4,218/109*2]

cut_z = [6,31]

fly_num = len(flyname_list)


## calculate G7f average
if not os.path.exists(save_path + '/figure/acc'):os.mkdir(save_path + '/figure/acc')     
acc = io.imread(data_source[0] + flyname_list[0][0] + '/figure/raw/trans_acc_DEEPCAD_C0_odor0_mean.tif')

for j in range(1):
    acc_ave = np.zeros_like(acc)
    for type in range(2):
        for fly_index in range(len(flyname_list[type])):
            ## load acc map
            acc = io.imread(data_source[type] + flyname_list[type][fly_index] + '/figure/raw/trans_acc_DEEPCAD_C1_odor' + str(j) + '_mean.tif')
            acc = acc / 65535
            acc[np.isinf(acc)] = np.nan
            acc[np.isnan(acc)] = 0
            acc_ave = acc_ave + acc
    acc = acc_ave / (len(flyname_list[0])+len(flyname_list[1]))
    acc_down = np.zeros((int(acc.shape[0]/voxel_size[0]),int(acc.shape[1]/voxel_size[1]),int(acc.shape[2]/voxel_size[2])))
    acc_z = np.zeros((int(acc.shape[1]/voxel_size[1]),int(acc.shape[2]/voxel_size[2])))
    for z in range(acc.shape[0]):
        acc_z = acc_z + cv2.resize(acc[z], (int(acc.shape[1]/voxel_size[1]),int(acc.shape[2]/voxel_size[2])), interpolation = cv2.INTER_AREA)
        if z % voxel_size[0] == voxel_size[0] - 1:
            acc_z = acc_z / voxel_size[0]
            acc_down[int(z/2)] = acc_z
            acc_z = np.zeros((int(acc.shape[1]/voxel_size[1]),int(acc.shape[2]/voxel_size[2])))
    acc_down = acc_down[cut_z[0]:cut_z[1],:,:]
    projection_3d((acc_down).transpose((1,2,0)),64*pixel_size[0],64*pixel_size[1],acc_down.shape[0]*pixel_size[2],30,'jet',[0,0.7],0,True,True,os.path.join(save_path + '/figure/acc', 'acc_DEEPCAD_C1_odor' + str(j) + '_mean.'))
