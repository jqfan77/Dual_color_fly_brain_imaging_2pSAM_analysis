##
## plot the dff0 traces
## the dff0 trace is averaged in selected brain regions
##

import numpy as np
import matplotlib.pylab as plt
import hdf5storage
import os
from skimage import io

## data
# flytype = 'rAch1h'
flytype = 'r5HT1.0'

date = '0513'

flynumber = '2'


## path
flyname = '2023'+date+'-nsyb-G7f-'+flytype+'/fly'+flynumber

result_path = '../results/3.odor_random_90_times_OCT_MCH_EA_new/'

if not os.path.exists(result_path + flyname + '/figure/region_voxel_trace'):os.mkdir(result_path + flyname + '/figure/region_voxel_trace')
file_start = 12
file_num = 9 ## file num
file_frame = 200


## load stim mat
stim_mat = hdf5storage.loadmat(result_path + flyname +'/Process/data_rearrange/stim_info.mat')['stim_mat']
stim_mat = stim_mat[0]
stim_mat = stim_mat[(file_start-1) * file_frame:(file_start+file_num-1) * file_frame]
stim_mat[stim_mat > 0.5] = 1
stim_mat_start = np.zeros_like(stim_mat)
stim_mat_end = np.zeros_like(stim_mat)
for i in range(stim_mat.shape[0]):
    if i == stim_mat.shape[0]-1:
        break
    if stim_mat[i] == 0 and stim_mat[i+1] == 1:
        stim_mat_start[i+1] = 1
    if stim_mat[i] == 1 and stim_mat[i+1] == 0:
        stim_mat_end[i+1] = 1
start = np.where(stim_mat_start==1)[0]
end = np.where(stim_mat_end==1)[0]
stimnum = start.shape[0]


## stim
stim_random_list_1 = [3,1,2,3,2,1,3,1,2,1,2,3,1,2,3,2,3,1,3,2,1,2,3,1,2,3,1,2,1,3,2,3,1,2,3,1,3,1,2,3,1,2,3,2,1,2,1,3,2,1,3,1,2,3,1,2,3,2,1,3,1,2,3,2,1,3,1,3,2,3,2,1,3,2,1,3,1,2,3,2,1,3,1,2,3,2,1,2,3,1];
stim_random_list_2 = [3,1,2,3,1,2,3,2,1,2,3,1,2,1,3,2,1,3,1,3,2,3,1,2,1,2,3,2,3,1,2,3,1,3,2,1,2,3,1,2,1,3,1,2,3,2,3,1,2,1,3,1,3,2,3,1,2,1,2,3,2,1,3,1,2,3,2,3,1,3,1,2,1,3,2,1,3,2,3,1,2,3,2,1,2,1,3,1,2,3];
stim_random_list_1.extend(stim_random_list_2)
stim_random_list = np.array(stim_random_list_1)
color_style = ['royalblue','firebrick','darkorange']
color = []
for i in range(stim_random_list.shape[0]):
    color.append(color_style[stim_random_list[i]-1])


## load and plot dff0
for root, dirs, files in os.walk(result_path + flyname + '/data/region_voxel_trace'):
    for f in files:
        if 'ave_voxel' in f and 'C2' in f and 'dff0' in f:
            trace_C2_path = os.path.join(root, f)
            trace_C2 = np.load(trace_C2_path)
            plt.figure()
            plt.plot(trace_C2,linewidth =0.3)
            for i in range(stimnum):
                if 'dff0' in f:
                    x = [start[i],start[i],end[i],end[i]]
                    y = [0.1, -0.1, -0.1, 0.1]
                    plt.fill(x, y, color = color[i], linewidth=0)
            plt.ylim(-0.5,2)
            plt.savefig(result_path+flyname+'/figure/region_voxel_trace/' + f.replace('.npy','') +'.png',dpi = 300)
            plt.savefig(result_path+flyname+'/figure/region_voxel_trace/' + f.replace('.npy','') +'.eps',dpi = 300)
            plt.close()
            print('Saved:', result_path, flyname, '/figure/region_voxel_trace/' + f.replace('.npy','') +'.png')
            print('Saved:', result_path, flyname, '/figure/region_voxel_trace/' + f.replace('.npy','') +'.eps')

        if 'ave_voxel' in f and 'C3' in f and 'dff0' in f:
            trace_C3_path = os.path.join(root, f)
            trace_C3 = np.load(trace_C3_path)
            plt.figure()
            plt.plot(trace_C3,linewidth =0.3)
            for i in range(stimnum):
                if 'dff0' in f:
                    x = [start[i],start[i],end[i],end[i]]
                    y = [0.1, -0.1, -0.1, 0.1]
                    plt.fill(x, y, color = color[i], linewidth=0)
            plt.ylim(-0.5,2)
            plt.savefig(result_path+flyname+'/figure/region_voxel_trace/' + f.replace('.npy','') +'.png',dpi = 300)
            plt.savefig(result_path+flyname+'/figure/region_voxel_trace/' + f.replace('.npy','') +'.eps',dpi = 300)
            plt.close()
            print('Saved:', result_path, flyname, '/figure/region_voxel_trace/' + f.replace('.npy','') +'.png')
            print('Saved:', result_path, flyname, '/figure/region_voxel_trace/' + f.replace('.npy','') +'.eps')