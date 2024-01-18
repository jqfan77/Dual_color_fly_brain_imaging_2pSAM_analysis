## 
## this code is used for calculate and plot acc map for each indicator
## realign should be down in Fiji!
##

import os
from skimage import io
import numpy as np
import pandas as pd 
from utils.projection_3d import *
import cv2

## realign should be down in Fiji!
realign = False


## path
# acc_source = './p5_representation_analysis/results/5HT-ver16/'
acc_source = './p5_representation_analysis/results/Ach-ver16/'


data_source =  '../results/3.odor_random_90_times_OCT_MCH_EA_new/nsyb-G7f-rAch1h/'
# data_source =  '../results/3.odor_random_90_times_OCT_MCH_EA_new/nsyb-G7f-r5HT1.0/'


flyname_list = [
           '20230417-fly2',
           '20230420-fly2',
           '20230420-fly3',
           '20230428-fly1',
           '20230507-fly1',
           '20230510-fly1',
           '20230510-fly2',
           '20230511-fly2',
           '20230511-fly3',
           '20230515-fly1',
           ]

# flyname_list = [
#            '20230429-r5HT1.0-fly1',
#            '20230506-r5HT1.0-fly1',
#            '20230513-r5HT1.0-fly1',
#            '20230513-r5HT1.0-fly2',
#            '20230516-r5HT1.0-fly2',
#            '20230516-r5HT1.0-fly4',
#            '20230517-r5HT1.0-fly1',
#            '20230601-r5HT1.0-fly1', 
#            '20230601-r5HT1.0-fly3',
#            '20230603-r5HT1.0-fly1',
#            ]


## parameter set
voxel_size = [2,4,4]

pixel_size = [318.51/256*4,318.51/256*4,218/109*2]

cut_z = [6,31]

fly_num = len(flyname_list)

if realign == False:
    # cal acc map
    for fly_index in range(len(flyname_list)):
        ## load acc
        acc = np.load(acc_source + flyname_list[fly_index] + '/Accuracy_map_DEEPCAD_formal/acc.npy')
        ## load atlas
        if 'Ach' in data_source:
            fly_name = flyname_list[fly_index].split('-')[0] + '-nsyb-G7f-rAch1h/' + flyname_list[fly_index].split('-')[1]
        else:
            fly_name = flyname_list[fly_index].split('-')[0] + '-nsyb-G7f-' + flyname_list[fly_index].split('-')[1]+'/'+flyname_list[fly_index].split('-')[2]
        atlas = io.imread(data_source + fly_name+'/align_to_atlas/Transformed_atlas.tif')

        if not os.path.exists(data_source + fly_name  + '/figure/raw'):os.mkdir(data_source + fly_name  + '/figure/raw')     

        ## save acc map tif
        for i in range(3):
            for j in range(1):
                io.imsave(data_source + fly_name + '/figure/raw/acc_DEEPCAD_C' + str(i) + '_odor' + str(j) + '_mean.tif', (acc[i,j].transpose((2,0,1))*65535).astype('uint16'))

if realign == True:
    ## plot acc map for each fly
    for fly_index in range(len(flyname_list)):
        if 'Ach' in data_source:
            fly_name = flyname_list[fly_index].split('-')[0] + '-nsyb-G7f-rAch1h/' + flyname_list[fly_index].split('-')[1]
        else:
            fly_name = flyname_list[fly_index].split('-')[0] + '-nsyb-G7f-' + flyname_list[fly_index].split('-')[1]+'/'+flyname_list[fly_index].split('-')[2]
        if not os.path.exists(data_source + fly_name + '/figure/acc'):os.mkdir(data_source + fly_name + '/figure/acc')
        ## load acc map
        for i in range(3):
            for j in range(1):
                acc = io.imread(data_source + fly_name + '/figure/raw/trans_acc_DEEPCAD_C' + str(i) + '_odor' + str(j) + '_mean.tif')
                acc = acc / 65535
                acc_down = np.zeros((int(acc.shape[0]/voxel_size[0]),int(acc.shape[1]/voxel_size[1]),int(acc.shape[2]/voxel_size[2])))
                acc_z = np.zeros((int(acc.shape[1]/voxel_size[1]),int(acc.shape[2]/voxel_size[2])))
                for z in range(acc.shape[0]):
                    acc_z = acc_z + cv2.resize(acc[z], (int(acc.shape[1]/voxel_size[1]),int(acc.shape[2]/voxel_size[2])), interpolation = cv2.INTER_AREA)
                    if z % voxel_size[0] == voxel_size[0] - 1:
                        acc_z = acc_z / voxel_size[0]
                        acc_down[int(z/2)] = acc_z
                        acc_z = np.zeros((int(acc.shape[1]/voxel_size[1]),int(acc.shape[2]/voxel_size[2])))
                acc_down = acc_down[cut_z[0]:cut_z[1],:,:]
                projection_3d(acc_down.transpose((1,2,0)),64*pixel_size[0],64*pixel_size[1],acc_down.shape[0]*pixel_size[2],30,'jet',[0,0.7],0,True,True,os.path.join(data_source + fly_name + '/figure/acc', 'acc_DEEPCAD_C' + str(i) + '_odor' + str(j) + '_mean.'))
                
    ## plot acc map for fly average
    if not os.path.exists(data_source + '/figure/acc'):os.mkdir(data_source + '/figure/acc')     
    for j in range(1):
        acc_down_C = []
        for i in range(3):
            acc_ave = np.zeros_like(acc)
            for fly_index in range(len(flyname_list)):
                if 'Ach' in data_source:
                    fly_name = flyname_list[fly_index].split('-')[0] + '-nsyb-G7f-rAch1h/' + flyname_list[fly_index].split('-')[1]
                else:
                    fly_name = flyname_list[fly_index].split('-')[0] + '-nsyb-G7f-' + flyname_list[fly_index].split('-')[1]+'/'+flyname_list[fly_index].split('-')[2]
                # load acc map
                acc = io.imread(data_source + fly_name + '/figure/raw/trans_acc_DEEPCAD_C' + str(i) + '_odor' + str(j) + '_mean.tif')
                acc = acc / 65535
                acc[np.isinf(acc)] = np.nan
                acc[np.isnan(acc)] = 0
                acc_ave = acc_ave + acc
            acc = acc_ave / len(flyname_list)
            acc_down = np.zeros((int(acc.shape[0]/voxel_size[0]),int(acc.shape[1]/voxel_size[1]),int(acc.shape[2]/voxel_size[2])))
            acc_z = np.zeros((int(acc.shape[1]/voxel_size[1]),int(acc.shape[2]/voxel_size[2])))
            for z in range(acc.shape[0]):
                acc_z = acc_z + cv2.resize(acc[z], (int(acc.shape[1]/voxel_size[1]),int(acc.shape[2]/voxel_size[2])), interpolation = cv2.INTER_AREA)
                if z % voxel_size[0] == voxel_size[0] - 1:
                    acc_z = acc_z / voxel_size[0]
                    acc_down[int(z/2)] = acc_z
                    acc_z = np.zeros((int(acc.shape[1]/voxel_size[1]),int(acc.shape[2]/voxel_size[2])))
            acc_down = acc_down[cut_z[0]:cut_z[1],:,:]
            projection_3d(acc_down.transpose((1,2,0)),64*pixel_size[0],64*pixel_size[1],acc_down.shape[0]*pixel_size[2],30,'jet',[0,0.7],0,True,True,os.path.join(data_source + '/figure/acc', 'acc_DEEPCAD_C' + str(i) + '_odor' + str(j) + '_mean.'))
            acc_down_C.append(acc_down)
        projection_3d((acc_down_C[0]-acc_down_C[1]).transpose((1,2,0)),64*pixel_size[0],64*pixel_size[1],acc_down.shape[0]*pixel_size[2],30,'jet',[0,0.15],0,True,True,os.path.join(data_source + '/figure/acc', 'acc_DEEPCAD_C0-C1_odor' + str(j) + '_mean.'))
