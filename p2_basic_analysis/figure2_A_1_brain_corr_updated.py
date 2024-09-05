## 
## plot the correlation map of each indicator
##

import os
from skimage import io
import numpy as np
from utils.projection_3d import *

realign = True # realign in Fiji

## path
corr_source_ach = '../pipeline-revision/results/nsyb-G7f-rAch1h/figures-for-revision1/mean_response_final_trial/'
corr_source_5ht = '../pipeline-revision/results/nsyb-G7f-r5HT1.0/figures-for-revision1/mean_response_final_trial/'

data_source_ach =  '../data/'
data_source_5ht =  '../data/'
data_source = '../data/'

flynames_ach = [
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
           ] 

flynames_5ht = [
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
           ]


## parameter set
cut_z = [28,130]
pixel_size = [318.51/512,318.51/512 ,218/218]

corr_ach = np.load(corr_source_ach + 'mean_response_corr_map.npy')
corr_5ht = np.load(corr_source_5ht + 'mean_response_corr_map.npy')
corr_ach[np.isinf(corr_ach)] = np.nan
corr_5ht[np.isinf(corr_5ht)] = np.nan

if realign == False:
    ## cal corr map
    for fly_index in range(len(flynames_ach)):
        ## load atlas
        fly_name = flynames_ach[fly_index]

        ## load corr
        corr_fly = corr_ach[fly_index]
        if not os.path.exists(data_source_ach + fly_name  + '/figure/raw'):os.mkdir(data_source_ach + fly_name  + '/figure/raw')     
        
        ## save corr map tif
        for j in range(2):
            io.imsave(data_source_ach + fly_name + '/figure/raw/response_corr_C' + str(j) + '.tif', (corr_fly[j].transpose((2,0,1))*65535).astype('uint16'))
    for fly_index in range(len(flynames_5ht)):
        ## load atlas
        fly_name = flynames_5ht[fly_index]
        ## load corr
        corr_fly = corr_5ht[fly_index]
        if not os.path.exists(data_source_5ht + fly_name  + '/figure/raw'):os.mkdir(data_source_5ht + fly_name  + '/figure/raw')     
        ## save corr map tif
        for j in range(2):
            io.imsave(data_source_5ht + fly_name + '/figure/raw/response_corr_C' + str(j) + '.tif', (corr_fly[j].transpose((2,0,1))*65535).astype('uint16'))

if realign == True:
    # plot corr map
    for fly_index in range(len(flynames_ach)):
        fly_name = flynames_ach[fly_index]
        
        if not os.path.exists(data_source_ach + fly_name + '/figure/re_corr'):
            os.mkdir(data_source_ach + fly_name + '/figure/re_corr')
        ## load corr map
        for i in range(2):
            
            corr = io.imread(data_source_ach + fly_name + '/figure/raw/Transformed_mean_response_corr_map_C' + str(i) + '.tif')
            corr[corr==0]=10000
            corr = corr / 10000-1
            corr[np.isnan(corr)] = 0
            corr_down = corr[cut_z[0]:cut_z[1],:,:]
            projection_3d(corr_down.transpose((1,2,0)),512*pixel_size[0],512*pixel_size[1],corr_down.shape[0]*pixel_size[2],30,None,[0,0.6],0,True,True,os.path.join(data_source_ach + fly_name + '/figure/re_corr', 'response_corr_C' + str(i) + '.'))
    
    for fly_index in range(len(flynames_ach)):
        fly_name = flynames_5ht[fly_index]
        
        if not os.path.exists(data_source_5ht + fly_name + '/figure/re_corr'):os.mkdir(data_source_5ht + fly_name + '/figure/re_corr')
        ## load corr map
        for i in range(2):
            corr = io.imread(data_source_5ht + fly_name + '/figure/raw/Transformed_mean_response_corr_map_C' + str(i) + '.tif')
            corr[corr==0]=10000
            corr = corr / 10000-1
            corr[np.isnan(corr)] = 0
            corr_down = corr[cut_z[0]:cut_z[1],:,:]
            projection_3d(corr_down.transpose((1,2,0)),512*pixel_size[0],512*pixel_size[1],corr_down.shape[0]*pixel_size[2],30,None,[0,0.6],0,True,True,os.path.join(data_source_5ht + fly_name + '/figure/re_corr', 'response_corr_C' + str(i) + '.'))
                

    if not os.path.exists(data_source + '/figure/re_corr'):os.mkdir(data_source + '/figure/re_corr')  
    if not os.path.exists(data_source_ach + '/nsyb-G7f-rAch1h/figure/re_corr'):os.mkdir(data_source_ach + '/nsyb-G7f-rAch1h/figure/re_corr')    
    if not os.path.exists(data_source_5ht + '/nsyb-G7f-r5HT1.0/figure/re_corr'):os.mkdir(data_source_5ht + '/nsyb-G7f-r5HT1.0/figure/re_corr')       

    corr = io.imread(data_source_ach + flynames_ach[0] + '/figure/raw/Transformed_mean_response_corr_map_C0.tif')
    corr_ave = np.zeros_like(corr)
    for fly_index in range(len(flynames_ach)):
        fly_name = flynames_ach[fly_index]
        ## load corr map
        corr = io.imread(data_source_ach + fly_name + '/figure/raw/Transformed_mean_response_corr_map_C0.tif')
        corr[corr==0]=10000
        corr = corr / 10000-1
        corr[np.isnan(corr)] = 0
        corr_ave = corr_ave + corr
    for fly_index in range(len(flynames_5ht)):
        fly_name = flynames_5ht[fly_index]
        ## load corr map
        corr = io.imread(data_source_5ht + fly_name + '/figure/raw/Transformed_mean_response_corr_map_C0.tif')
        corr[corr==0]=10000
        corr = corr / 10000-1
        corr[np.isnan(corr)] = 0
        corr_ave = corr_ave + corr
    corr = corr_ave / (len(flynames_ach) + len(flynames_5ht))
    corr_down = corr[cut_z[0]:cut_z[1],:,:]
    projection_3d(corr_down.transpose((1,2,0)),512*pixel_size[0],512*pixel_size[1],corr_down.shape[0]*pixel_size[2],30,None,[0,0.4],0,True,True,os.path.join(data_source + '/figure/re_corr', 'response_corr_max_C2.'))
    # projection_3d(corr_down.transpose((1,2,0)),512*pixel_size[0],512*pixel_size[1],corr_down.shape[0]*pixel_size[2],30,None,[0,1],1,True,True,os.path.join(data_source + '/figure/re_corr', 'response_corr_mean_C2.'))

    corr_ave = np.zeros_like(corr)
    for fly_index in range(len(flynames_ach)):
        fly_name = flynames_ach[fly_index]
        ## load corr map
        corr = io.imread(data_source_ach + fly_name + '/figure/raw/Transformed_mean_response_corr_map_C1.tif')
        corr[corr==0]=10000
        corr = corr / 10000-1
        corr[np.isnan(corr)] = 0
        corr_ave = corr_ave + corr
    corr = corr_ave / len(flynames_ach)
    corr_down = corr[cut_z[0]:cut_z[1],:,:]
    projection_3d(corr_down.transpose((1,2,0)),512*pixel_size[0],512*pixel_size[1],corr_down.shape[0]*pixel_size[2],30,None,[0,0.5],0,True,True,os.path.join(data_source_ach + '/nsyb-G7f-rAch1h/figure/re_corr', 'response_corr_max_C3.'))
    # projection_3d(corr_down.transpose((1,2,0)),512*pixel_size[0],512*pixel_size[1],corr_down.shape[0]*pixel_size[2],30,None,[0,1],1,True,True,os.path.join(data_source_ach + '/figure/re_corr', 'response_corr_mean_C3.'))
        
    corr_ave = np.zeros_like(corr)
    for fly_index in range(len(flynames_5ht)):
        fly_name = flynames_5ht[fly_index]
        ## load corr map
        corr = io.imread(data_source_5ht + fly_name + '/figure/raw/Transformed_mean_response_corr_map_C1.tif')
        corr[corr==0]=10000
        corr = corr / 10000-1
        corr[np.isnan(corr)] = 0
        corr_ave = corr_ave + corr
    corr = corr_ave / len(flynames_5ht)
    corr_down = corr[cut_z[0]:cut_z[1],:,:]
    projection_3d(corr_down.transpose((1,2,0)),512*pixel_size[0],512*pixel_size[1],corr_down.shape[0]*pixel_size[2],30,None,[0,0.15],0,True,True,os.path.join(data_source_5ht + '/nsyb-G7f-r5HT1.0/figure/re_corr', 'response_corr_max_C3.'))
    # projection_3d(corr_down.transpose((1,2,0)),512*pixel_size[0],512*pixel_size[1],corr_down.shape[0]*pixel_size[2],30,None,[0,1],1,True,True,os.path.join(data_source_5ht + '/figure/re_corr', 'response_corr_mean_C3.'))
        