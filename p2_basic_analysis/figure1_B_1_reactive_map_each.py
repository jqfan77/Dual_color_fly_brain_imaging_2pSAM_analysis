##
## calculate the response map of each fly
## calculate the sem of each trial, and average them
##

import numpy as np
from matplotlib.pyplot import MultipleLocator
import os
from skimage import io
from scipy import stats
from utils.projection_3d import *

## data
flynames = [
           'nsyb-G7f-rAch1h/20230417-nsyb-G7f-rAch1h/fly2', 
           'nsyb-G7f-rAch1h/20230420-nsyb-G7f-rAch1h/fly2',
           'nsyb-G7f-rAch1h/20230420-nsyb-G7f-rAch1h/fly3',
           'nsyb-G7f-rAch1h/20230428-nsyb-G7f-rAch1h/fly1', 
           'nsyb-G7f-rAch1h/20230507-nsyb-G7f-rAch1h/fly1', 
           'nsyb-G7f-rAch1h/20230510-nsyb-G7f-rAch1h/fly1',
           'nsyb-G7f-rAch1h/20230510-nsyb-G7f-rAch1h/fly2',
           'nsyb-G7f-rAch1h/20230511-nsyb-G7f-rAch1h/fly2', 
           'nsyb-G7f-rAch1h/20230511-nsyb-G7f-rAch1h/fly3',
           'nsyb-G7f-rAch1h/20230515-nsyb-G7f-rAch1h/fly1',
           ] 

# flynames = [
#            'nsyb-G7f-r5HT1.0/20230429-nsyb-G7f-r5HT1.0/fly1',
#            'nsyb-G7f-r5HT1.0/20230506-nsyb-G7f-r5HT1.0/fly1',
#            'nsyb-G7f-r5HT1.0/20230513-nsyb-G7f-r5HT1.0/fly1',
#            'nsyb-G7f-r5HT1.0/20230513-nsyb-G7f-r5HT1.0/fly2',
#            'nsyb-G7f-r5HT1.0/20230516-nsyb-G7f-r5HT1.0/fly2',
#            'nsyb-G7f-r5HT1.0/20230516-nsyb-G7f-r5HT1.0/fly4',
#            'nsyb-G7f-r5HT1.0/20230517-nsyb-G7f-r5HT1.0/fly1',
#            'nsyb-G7f-r5HT1.0/20230601-nsyb-G7f-r5HT1.0/fly1',
#            'nsyb-G7f-r5HT1.0/20230601-nsyb-G7f-r5HT1.0/fly3',
#            'nsyb-G7f-r5HT1.0/20230603-nsyb-G7f-r5HT1.0/fly1',
#            ]


## path
datapath = '../results/3.odor_random_90_times_OCT_MCH_EA_new/'


## stim mat
stim_random_list_1 = [3,1,2,3,2,1,3,1,2,1,2,3,1,2,3,2,3,1,3,2,1,2,3,1,2,3,1,2,1,3,2,3,1,2,3,1,3,1,2,3,1,2,3,2,1,2,1,3,2,1,3,1,2,3,1,2,3,2,1,3,1,2,3,2,1,3,1,3,2,3,2,1,3,2,1,3,1,2,3,2,1,3,1,2,3,2,1,2,3,1]
stim_random_list_2 = [3,1,2,3,1,2,3,2,1,2,3,1,2,1,3,2,1,3,1,3,2,3,1,2,1,2,3,2,3,1,2,3,1,3,2,1,2,3,1,2,1,3,1,2,3,2,3,1,2,1,3,1,3,2,3,1,2,1,2,3,2,1,3,1,2,3,2,3,1,3,1,2,1,3,2,1,3,2,3,1,2,3,2,1,2,1,3,1,2,3]
stim_random_list_1.extend(stim_random_list_2)
stim_random_list = np.array(stim_random_list_1)
stim_random_list = stim_random_list[3:180] ## cut the first three stim

for flyname in flynames:

    ## path
    savepath_raw = datapath + flyname + '/figure'
    if not os.path.exists(savepath_raw):os.mkdir(savepath_raw)

    savepath_raw = datapath + flyname + '/figure/raw'
    if not os.path.exists(savepath_raw):os.mkdir(savepath_raw)

    savepath = datapath + flyname + '/figure/response'
    if not os.path.exists(savepath):os.mkdir(savepath)

    ## parameter set
    if 'rAch1h' in flyname:
        channel_name = ['G7f','Ach']
    if 'r5HT1.0' in flyname:
        channel_name = ['G7f','5HT']
    odor_name = ['OCT','MCH','EA']

    ## atlas
    atlas = io.imread(datapath + flyname + '/align_to_atlas/Transformed_atlas.tif')

    # calculate mean and std
    for c in range(2,4):
        ## load dff0 trace
        if c == 2:
            dff0 = np.load(datapath + flyname + '/data/dff0_0-20_down1_C'+str(c)+'.npy')
        if c == 3:
            if 'rAch1h' in flyname:
                dff0 = np.load(datapath + flyname + '/data/dff0_0-20_down1_C'+str(c)+'.npy')
            if 'r5HT1.0' in flyname:
                dff0 = np.load(datapath + flyname + '/data/dff0_0-40_down2_C'+str(c)+'.npy')

        ## cut the voxel out of atlas
        for x in range(dff0.shape[3]):
            for y in range(dff0.shape[4]):
                for z in range(dff0.shape[2]):
                    index = atlas[z + 13,x,y]
                    if index == 0:
                        dff0[:,:,z,x,y] = 0

        ## cal the sem of each trial
        trial_std = np.zeros((dff0.shape[2],dff0.shape[3],dff0.shape[4]))
        dff0 = dff0[3:180]
        trace_trial_ave = np.mean(dff0,axis=0)
        trial_std = stats.sem(trace_trial_ave, axis = 0)
        
        ## save response map
        io.imsave(savepath_raw + '/response_C' + str(c) + '_trial_std.tif', (trial_std*65535).astype('uint16'))

