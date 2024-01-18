##
## This code calculate the dff0 for every voxel in the stim window
##

import numpy as np
import matplotlib.pylab as plt
import hdf5storage
import os

## data
flytype = 'rAch1h'
# flytype = 'r5HT1.0'

date = '0510'

flynumber = '2'

odor_id = [2,3] ## channel select

sw = False ## slide window
 

## parameter set
sw_frequency = 13 ## slide window frequency

voxel_size = [256,256,25] 

min_percent = 0.3 # f0 trunc

if flytype == 'r5HT1.0':
    sw_len_2 = [20,40] # stim window length
    time_downsample_2 = [1,2] # time downsample
    stim_before_2 = [0,0] # stim window start
if flytype == 'rAch1h':
    sw_len_2 = [20,20] 
    time_downsample_2 = [1,1] 
    stim_before_2 = [0,0] 

if sw:
    sw_len_2 = [item * sw_frequency for item in sw_len_2] 
    stim_before_2 = [item * sw_frequency for item in stim_before_2]
    trial_num = 18 # total trail num
    file_frame = 2600 # num of frame in each file
    frequency = 30
    dff0_sw = 200*13/(30) # dff0 calculate window
else:
    trial_num = 180
    file_frame = 200
    frequency = 30/13
    dff0_sw = 200/(30/13) 
    

## load path
flyname = 'nsyb-G7f-'+flytype+'/2023'+date+'-nsyb-G7f-'+flytype+'/fly'+flynumber

result_path = '../results/3.odor_random_90_times_OCT_MCH_EA_new/'

if sw:
    tracepath = result_path + flyname + '/data/trace_sw/'
    tracename = '2023'+date+'-nsyb-G7f-'+flytype+'-fly'+flynumber+'-sw-C'
else:
    tracepath = result_path + flyname + '/data/trace/'
    tracename = '2023'+date+'-nsyb-G7f-'+flytype+'-fly'+flynumber+'-C'


## save path
if sw:
    savename = ['dff0_'+str(-stim_before_2[0])+'-'+str(-stim_before_2[0]+sw_len_2[0])+'_down'+str(time_downsample_2[0])+'_sw_C','dff0_'+str(-stim_before_2[1])+'-'+str(-stim_before_2[1]+sw_len_2[1])+'_down'+str(time_downsample_2[1])+'_sw_C']
    print('savename:',savename[0],savename[1])
else:
    savename = ['dff0_'+str(-stim_before_2[0])+'-'+str(-stim_before_2[0]+sw_len_2[0])+'_down'+str(time_downsample_2[0])+'_C','dff0_'+str(-stim_before_2[1])+'-'+str(-stim_before_2[1]+sw_len_2[1])+'_down'+str(time_downsample_2[1])+'_C']
    print('savename:',savename[0],savename[1])


## file load info
if sw:
    files = os.listdir(tracepath + tracename + '2')
    file_start = 100 
    for i in range( len(files)):
        if int((files[i].split('_')[1]).split('.')[0]) < file_start:
            file_start = int((files[i].split('_')[1]).split('.')[0])
    file_num = len(files)
else:
    files = os.listdir(tracepath + tracename + '2')
    file_start = 1 
    file_num = len(files)
print('file load from:', file_start,'to ', file_start + file_num)


## load stim mat
if sw:
    stim_mat = hdf5storage.loadmat(result_path + flyname +'/Process/data_rearrange/stim_info_sw.mat')['stim_mat']
else:
    stim_mat = hdf5storage.loadmat(result_path + flyname +'/Process/data_rearrange/stim_info.mat')['stim_mat']

stim_mat = stim_mat[0]
stim_mat[stim_mat > 0.5] = 1
stim_mat_start_2 = np.zeros_like(stim_mat)
for i in range(stim_mat.shape[0]):
    if i == stim_mat.shape[0]-1:
        break
    if stim_mat[i] == 0 and stim_mat[i+1] == 1:
        stim_mat_start_2[i+1] = 1


## calculate dff0
for odor in range(odor_id[0],odor_id[1]):

    stim_before = stim_before_2[odor-2]
    sw_len = sw_len_2[odor-2]
    time_downsample = time_downsample_2[odor-2]

    if stim_before >= 0:
        stim_mat_start = stim_mat_start_2[stim_before:stim_mat_start_2.shape[0]]
    else:
        stim_mat_start = np.concatenate((np.zeros(-stim_before),stim_mat_start_2))
    stim_mat_start = stim_mat_start[(file_start-1)*file_frame:stim_mat_start.shape[0]]
    start = np.where(stim_mat_start==1)[0]
    stimnum = start.shape[0]
    
    recent_stim = 0
    trail_whole = []

    ## read file
    for i in range(file_num):
        trace = []
        
        stim_start_part = stim_mat_start[i*file_frame:(i+1)*file_frame]

        ## if no stim, continue
        if np.sum(stim_start_part) == 0 and np.sum(stim_mat_start[(i+1)*file_frame:(i+2)*file_frame]) == 0:
            continue
      
        ## if next trace began to start stim, load trace
        if np.sum(stim_start_part) == 0 and np.sum(stim_mat_start[(i+1)*file_frame:(i+2)*file_frame]) != 0:
            trace_right = np.load(tracepath + tracename + str(odor) + '/Trace_%04d.npy'%(i + file_start))
            print("file loaded:" + tracepath + tracename + str(odor) + '/Trace_%04d.npy'%(i + file_start))
            loaded_file = i
            continue 

        ## check if next trace have loaded
        trace_before = trace_right
        trace.append(trace_before)
        if loaded_file == i:
            trace_right = trace_next
        else:
            trace_right = np.load(tracepath + tracename + str(odor) + '/Trace_%04d.npy'%(i + file_start)) 
            print("file loaded:" + tracepath + tracename + str(odor) + '/Trace_%04d.npy'%(i + file_start))
            loaded_file = i
        trace.append(trace_right)

        ## check if need to load next trace
        for index in range(file_frame-1,-1,-1):
            if stim_start_part[index] == 1:
                last_stim_index = index
                break
        
        ## load next trace
        if last_stim_index + sw_len > file_frame - 1:
            trace_next = np.load(tracepath + tracename + str(odor) + '/Trace_%04d.npy'%(i + file_start + 1)) 
            print("file loaded:" + tracepath + tracename + str(odor) + '/Trace_%04d.npy'%(i + file_start + 1))
            loaded_file = i + 1
            trace.append(trace_next)
        
        ## concate trace
        trace = np.stack(trace) 
        trace = trace.reshape(-1,voxel_size[2],voxel_size[0],voxel_size[1])

        ## stim start part
        start_part = np.where(stim_start_part==1)[0]
        stimnum_part = start_part.shape[0]
        start_part = start_part + file_frame
        
        ## calculate dff0
        for st_num in range(stimnum_part):
            recent_stim = recent_stim + 1
            if recent_stim > trial_num:
                break

            trail = np.zeros((int(sw_len/time_downsample),voxel_size[2],voxel_size[0],voxel_size[1]))
            if sw: ## if slide window, calculate one f0 for each trial
                trace_win = np.sort(trace[int(start_part[st_num] -frequency*dff0_sw):(start_part[st_num]),:,:,:],axis=0)
                f0 = np.mean(trace_win[0:int(min_percent*frequency*dff0_sw)],0)
            for t in range(0,sw_len,time_downsample):
                if not sw: ## if not slide window, calculate one f0 for each stim
                    trace_win = np.sort(trace[int(start_part[st_num] -frequency*dff0_sw + t):(start_part[st_num] + t),:,:,:],axis=0)
                    f0 = np.mean(trace_win[0:int(min_percent*frequency*dff0_sw)],0)
                trail[int(t/time_downsample),:,:,:] = (trace[start_part[st_num] + t,:,:,:]-f0)/f0
            
            trail_whole.append(trail)
           
            print('recent_stim:',recent_stim)
        
        if recent_stim > trial_num:
            break

    print('total num of stim:',len(trail_whole))

    ## concate trials
    trail_whole = np.stack(trail_whole)
    trail_whole[np.isnan(trail_whole)] = 0
    trail_whole[np.isinf(trail_whole)] = 0
 
    ## save dff0
    if not os.path.exists(result_path + flyname + '/data'):os.mkdir(result_path + flyname + '/data')
    np.save(result_path + flyname + '/data/'+ savename[odor-2] + str(odor) + '.npy',trail_whole)

