##
## plot the hotmap of trial response
## for two flys choosed, average the first several trials of three odors separately in several brain region for 3 plombs
## the voxels in each several brain region is randomly choosed and each voxel is not zero
## the first stimulation is not involved becasue of the high response
##

import numpy as np
import matplotlib.pylab as plt
import hdf5storage
import os
from skimage import io
import pandas as pd
from matplotlib.pylab import mpl
import sys

## path
flyname_Ach = '20230420-nsyb-G7f-rAch1h/fly2'
flyname_5HT = '20230513-nsyb-G7f-r5HT1.0/fly2'

result_path = '../results/3.odor_random_90_times_OCT_MCH_EA_new/'

atlaspath = '/align_to_atlas'

c_names =['nsyb-G7f-rAch1h/','nsyb-G7f-r5HT1.0/']


## parameter set
brain_region_choose = [64,65,66,55,72,73,74,63,23,26] ## choose brain region
trial_num_ave = 30 ## average 30 trial
sw_flag = False ## slide window?
point_num = 10 # random voxel num in every brain region
frequence = 30/13
region_sort = [4,5,6,2,7,8,9,3,0,1]


## load atlas region name
brain_region = pd.read_excel('p4_figures/utils/mmc2_brief_adapted20220609.xlsx')
brain_region_index = brain_region['Brain Atlas Index']
brain_region_name = brain_region['Full Region Name']
select_name = []


## print choosed brain region
for i in range(len(brain_region_choose)):
    select_index = np.where(brain_region_index == brain_region_choose[i])
    select_name.append(brain_region_name[select_index[0][0]])
    print('select_brain_region:', brain_region_name[select_index[0][0]])


## main
dff0_sort_2 = []
for c_name in c_names:
    
    if '-rAch1h' in c_name:
        flyname = flyname_Ach
    if '-r5HT1.0' in c_name:
        flyname = flyname_5HT

    ## trace path
    tracepath = result_path + c_name + flyname + '/data/trace/'

    ## load atlas
    atlas = io.imread(result_path+ c_name + flyname+atlaspath+'/Transformed_atlas.tif')

    ## load stim mat
    stim_random_list_1 = [3,1,2,3,2,1,3,1,2,1,2,3,1,2,3,2,3,1,3,2,1,2,3,1,2,3,1,2,1,3]
    stim_random_list = np.array(stim_random_list_1)
    stim_mat = hdf5storage.loadmat(result_path + c_name +  flyname +'/Process/data_rearrange/stim_info.mat')['stim_mat']
    stim_mat = stim_mat[0]
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

    ## load trace
    all_exist = 0 ## all dff0 exist?
    dff0_c2 = [] ## dff0 cut 2 plombs
    for c in range(2,4):

        ## create save path
        if not os.path.exists(result_path + c_name +  flyname + '/data/dff0_cut_region_odor'):os.mkdir(result_path + c_name +  flyname + '/data/dff0_cut_region_odor')
            
        ## load dff0 (if every cut dff0 exist, do not need load the whole dff0)
        for i in range(len(brain_region_choose)):
            if not os.path.exists(result_path + c_name +  flyname + '/data/dff0_cut_region_odor/dff0_C'+str(c-2)+'_cut_'+str(brain_region_choose[i]) + '.npy'):
                all_exist = 0
                if c == 2:
                    dff0 = np.load(result_path + c_name +  flyname + '/data/dff0_-3-20_down1_C'+str(c)+'.npy')
                if c == 3:
                    if '-rAch1h' in c_name:
                        dff0 = np.load(result_path + c_name +  flyname + '/data/dff0_-3-20_down1_C'+str(c)+'.npy')
                    if '-r5HT1.0' in c_name:
                        dff0 = np.load(result_path + c_name +  flyname + '/data/dff0_-6-40_down2s_C'+str(c)+'.npy')
                dff0_cut = dff0[3:3+trial_num_ave] ## the fist stim is not involved
                dff0_c2.append(dff0_cut)
                break
            all_exist = 1
    print('all_exist:',all_exist)

    ## load cut dff0 (if not exist, sample from the whole dff0)
    dff0_cut_voxel_all = [] ## dff0 voxel (every odor, every fly)
    for c in range(2):
        dff0_cut_voxel_all.append([])

    for i in range(len(brain_region_choose)):
        if all_exist == 1:
            ## if all dff0 already exist, just load them
            for c in range(2,4):
                dff0_cut_voxel_all[c-2].append(np.load(result_path + c_name +  flyname + '/data/dff0_cut_region_odor/dff0_C'+str(c-2)+'_cut_'+str(brain_region_choose[i]) + '.npy'))
        else:
            ## find voxels in every brain region
            points = np.where(atlas == brain_region_choose[i])

            ## choose the voxels in the cut region
            points_in = []
            for p in range(len(points[0])):
                if (points[0][p] - 13 ) < 25:
                    points_in.append([points[0][p],points[1][p],points[2][p]])
            print(str(len(points_in)) + ' voxels in area ' + str(brain_region_choose[i]))

            ## if the number of voxels is not enough, break
            if len(points_in) < point_num:
                print('no area in ' + str(brain_region_choose[i]))
            
            ## check is each voxel has response
            good_voxels = 0
            iter = 0
            while good_voxels == 0:
                iter = iter + 1
                if iter > 2000:
                    print('error')
                    sys.exit()
                good_voxels = 1

                ## random sample some point in voxels
                a = np.random.randint(0,len(points_in),point_num)
                for point_index in range(point_num):
                    point_choose = [points_in[a[point_index]][0],points_in[a[point_index]][1],points_in[a[point_index]][2]]
                    if np.sum(dff0_c2[0][:,:,point_choose[0] - 13,point_choose[1],point_choose[2]]) < 5 or  np.sum(dff0_c2[0][:,:,point_choose[0] - 13,point_choose[1],point_choose[2]]) > 300:
                        good_voxels = 0
                        break
            
            ## chooed voxels
            for c in range(2):
                dff0_cut_voxel = [] ## dff0(every odor, every fly, every voxel)
                for point_index in range(point_num):
                    point_choose = [points_in[a[point_index]][0],points_in[a[point_index]][1],points_in[a[point_index]][2]]
                    dff0_cut_voxel.append(dff0_c2[c][:,:,point_choose[0] - 13,point_choose[1],point_choose[2]])
                dff0_cut_voxel_all[c].append(dff0_cut_voxel)

                ## save sampled voxels in every brain region
                np.save(result_path + c_name +  flyname + '/data/dff0_cut_region_odor/dff0_C'+str(c)+'_cut_'+str(brain_region_choose[i]) + '.npy',dff0_cut_voxel)

    ## reshape and append every flys dff0
    dff0_cut_voxel_all = np.stack(dff0_cut_voxel_all)
    dff0_cut_voxel_all = dff0_cut_voxel_all.reshape((2,-1,dff0_cut_voxel_all.shape[3],dff0_cut_voxel_all.shape[4]))

    ## separate each odor
    trace_odor = []
    for c in range(2):
        trace_odor.append([])
        for odor in range(1,4):
            ## extract odor trial
            index = np.where(stim_random_list == odor)
            trace_odor[c].append(dff0_cut_voxel_all[c,:,index,:])
    trace_odor = np.stack(trace_odor)    
    trace_odor = trace_odor[:,:,0,:,:,:]
    trace_odor[np.isinf(trace_odor)] = np.nan
    trace_odor = np.nanmean(trace_odor,2)
    
    ## sort the first odor
    dff0_sort = []
    sort_index = []
    for i in range(len(brain_region_choose)):
        dff0_single_region = trace_odor[0,0][i*point_num:(i+1)*point_num]
        dff0_single_region = np.sum(dff0_single_region,1)
        sort_index.append(np.argsort(-dff0_single_region, kind='quicksort', order=None))

    ## sort the others by the first odor
    for c in range(2):
        dff0_sort.append([])
        for i in range(3):
            dff0_sort[c].append([])
            for j in range(len(brain_region_choose)):
                dff0_single_region = trace_odor[c,i,j*point_num:(j+1)*point_num]
                dff0_sort[c][i].extend(dff0_single_region[sort_index[j]])

    dff0_sort_2.extend(dff0_sort)

dff0_sort = np.stack(dff0_sort_2)

## plot and save
plt.figure(figsize=(15,15))
for c in range(4):
    for odor in range(3):
        if c == 0 or c == 2:
            cmap = plt.get_cmap('Greens')
        if c == 1:
            cmap = plt.get_cmap('Purples')
        if c == 3:
            cmap = plt.get_cmap('Blues')
        plt.subplot(1,3*4,c*3 + odor+1)
        plt.imshow(dff0_sort[c,odor],vmin=0,vmax=0.5,cmap=cmap)
        plt.xticks([0, dff0_sort.shape[3]-1],[0,np.around((dff0_sort.shape[3]-1)/frequence,2)])
        plt.colorbar()

mpl.rcParams['ps.fonttype'] = 42

plt.savefig(result_path + '/figure/response/hotmap_odor.png',dpi = 300,bbox_inches = 'tight')
plt.savefig(result_path + '/figure/response/hotmap_odor.eps',dpi = 300,bbox_inches = 'tight')

plt.close()