##
## This code calculate the dff0 for every voxel before the first stimulation
##

import numpy as np
import matplotlib.pyplot as plt
import os
import hdf5storage
import time

## data
flytype = 'rAch1h'
# flytype = 'r5HT1.0'

date = '0507'

flynumber = '1'


## load path
result_path = '../results/3.odor_random_90_times_OCT_MCH_EA_new/'

tracename = '2023'+date+'-nsyb-G7f-'+flytype+'-fly'+flynumber+'-C'

flyname = 'nsyb-G7f-'+flytype+'/2023'+date+'-nsyb-G7f-'+flytype+'/fly'+flynumber

trace_path = result_path + flyname + '/data/trace/'


## parameter set
sw = 35 # one stim in real time
voxel_size = [256,256,25] 
file_frame = 200 # num of frame in each file
frequency = 30/13 
dff0_sw = 200/(30/13) # dff0 calculate window
min_percent = 0.3 # f0 trunc
trace_len = int(frequency*sw*10) ## trace length

if flytype == 'r5HT1.0':
    time_downsample = 2 ## time downsample
if flytype == 'rAch1h':
    time_downsample = 1


## load stim mat
stim_mat = hdf5storage.loadmat(result_path + flyname +'/Process/data_rearrange/stim_info.mat')['stim_mat']
stim_mat = stim_mat[0]
for i in range(stim_mat.shape[0]):
    if stim_mat[i] != 0:
        break
start_a = i - trace_len
start_b = i - 1


## extract start and pre
for odor in range(2,4):

    ## read files
    trace = []
    file_a = int(int(start_a-dff0_sw*frequency)/file_frame)
    file_b = int(start_b/file_frame)+1
    print('file load from:', file_a,'to ', file_b)

    for index in range(file_a,file_b):
        trace_raw = np.load(trace_path + tracename + str(odor) + '/Trace_%04d.npy'%(index))
        trace.append(trace_raw)
        print('loaded:' + trace_path + flyname + '-c' + str(odor) + '/Trace_%04d.npy'%(index))

    ## concate trace
    trace = np.stack(trace)
    trace = trace.reshape((-1,voxel_size[2],voxel_size[0],voxel_size[1]))

    ## cal dff0
    if trace_len%time_downsample == 0:
        trace_start = np.zeros((int(trace_len/time_downsample),voxel_size[2],voxel_size[0],voxel_size[1]))
    if trace_len%time_downsample == 1:
        trace_start = np.zeros((int(trace_len/time_downsample)+1,voxel_size[2],voxel_size[0],voxel_size[1]))
    for i in range(0,trace_len,time_downsample):
        trace_win = np.sort(trace[int(start_a - file_a*file_frame + i -frequency*dff0_sw):(start_a - file_a*file_frame + i),:,:,:], axis=0)
        f0 = np.mean(trace_win[0:int(min_percent*frequency*dff0_sw)],0)
        trace_start[int(i/time_downsample),:,:,:] = (trace[start_a - file_a*file_frame + i,:,:,:]-f0)/f0
    
    ## zero the nan and inf
    print('caculating finished!')
    trace_start[np.isnan(trace_start)] = 0
    trace_start[np.isinf(trace_start)] = 0
  
    ## save dff0
    if not os.path.exists(result_path + flyname + '/data'):os.mkdir(result_path + flyname + '/data')
    np.save(result_path + flyname +'/data/dff0_start_long_c'+str(odor)+'.npy',trace_start)
    print('trace saved!')