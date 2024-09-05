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

def compute_auc(trace,fps = 30/13):
    auc = 0
    num_tp = len(trace)
    for i in range(num_tp-1):
        auc = auc+(trace[i]+trace[i+1])*(1/fps)/2
    return auc
def compute_auc_dff0_trial_avg(data,fps = 30/13):
    size_x = np.size(data,2)
    size_y = np.size(data,3)
    size_z = np.size(data,1)
    num_tp = np.size(data,0)
    auc = np.zeros((size_z,size_x,size_y))
    for i in range(num_tp-1):
        auc = auc+(data[i,:,:,:]+data[i+1,:,:,:])*(1/fps)/2
    return auc
def compute_auc_dff0(data,fps = 30/13):
    size_x = np.size(data,3)
    size_y = np.size(data,4)
    size_z = np.size(data,2)
    num_tp = np.size(data,1)
    num_trial = np.size(data,0)
    auc = np.zeros((num_trial,size_z,size_x,size_y))
    for i in range(num_tp-1):
        auc = auc+(data[:,i,:,:,:]+data[:,i+1,:,:,:])*(1/fps)/2
    return auc


## data
flynames = [
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
datapath = '../data/'


## stim mat
stim_random_list_1 = [3,1,2,3,2,1,3,1,2,1,2,3,1,2,3,2,3,1,3,2,1,2,3,1,2,3,1,2,1,3,2,3,1,2,3,1,3,1,2,3,1,2,3,2,1,2,1,3,2,1,3,1,2,3,1,2,3,2,1,3,1,2,3,2,1,3,1,3,2,3,2,1,3,2,1,3,1,2,3,2,1,3,1,2,3,2,1,2,3,1]
stim_random_list_2 = [3,1,2,3,1,2,3,2,1,2,3,1,2,1,3,2,1,3,1,3,2,3,1,2,1,2,3,2,3,1,2,3,1,3,2,1,2,3,1,2,1,3,1,2,3,2,3,1,2,1,3,1,3,2,3,1,2,1,2,3,2,1,3,1,2,3,2,3,1,3,1,2,1,3,2,1,3,2,3,1,2,3,2,1,2,1,3,1,2,3]
stim_random_list_1.extend(stim_random_list_2)
stim_random_list = np.array(stim_random_list_1)
stim_random_list = stim_random_list[3:180] ## cut the first three stim

for flyname in flynames:

    print(flyname)

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
        print('c'+str(c))
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

        ## cal the std of trial average
        trial_std = np.zeros((dff0.shape[2],dff0.shape[3],dff0.shape[4]))
        dff0 = dff0[3:180]

        #### result avg
        trace_trial_ave = stats.sem(dff0,axis=1)
        trial_sem = np.mean(trace_trial_ave, axis = 0)
        trace_trial_ave = np.std(dff0, axis = 1,ddof = 1)
        trial_std = np.mean(trace_trial_ave, axis = 0)
        trace_trial_ave = np.mean(dff0,axis = 1)
        trial_avg = np.mean(trace_trial_ave,axis = 0)
        ## cal AUC
        if c==3 and 'r5HT1.0' in flyname:
            fps = 30/13/2
        else:
            fps = 30/13
        auc = compute_auc_dff0(dff0,fps)
        auc = np.mean(auc,axis = 0)
        # save response map
        io.imsave(savepath_raw + '/response_C' + str(c) + '_std_1.tif', (trial_std*65535).astype('uint16'))
        io.imsave(savepath_raw + '/response_C' + str(c) + '_sem_1.tif', (trial_sem*65535).astype('uint16'))
        io.imsave(savepath_raw + '/response_C' + str(c) + '_auc_1.tif', (auc/10*65535).astype('uint16'))
        io.imsave(savepath_raw + '/response_C' + str(c) + '_avg_1.tif', (trial_avg*65535).astype('uint16'))

