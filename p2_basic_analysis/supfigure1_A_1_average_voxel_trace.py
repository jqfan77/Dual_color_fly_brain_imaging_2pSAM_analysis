##
## calculate averaged dff0 trace in selected brain regions
##

import numpy as np
import hdf5storage
import os
from skimage import io
import pandas as pd
import ipdb

## data
odor_name = 'nsyb-G7f-rAch1h/'
# odor_name = 'nsyb-G7f-r5HT1.0/'

flynames = ['20230420-nsyb-G7f-rAch1h/fly2']
# flynames = ['20230513-nsyb-G7f-r5HT1.0/fly2']


## path
datapath = '../results/3.odor_random_90_times_OCT_MCH_EA_new/'

atlaspath = '/align_to_atlas'

result_path = '../results/3.odor_random_90_times_OCT_MCH_EA_new/'


## parameter set
file_frame = 200 ## num of frame
file_start = 12
file_num = 9 ## file num
frequency = 30/13
dff0_sw = 200/(30/13) # dff0 calculate window
min_percent = 0.3 # f0 trunc
brain_region_choose = [7,18,19,55,65,66] ## choose brain region


## load atlas region name
brain_region = pd.read_excel('p4_base_analysis/utils/mmc2_brief_adapted20220609.xlsx')
brain_region_index = brain_region['Brain Atlas Index']
brain_region_name = brain_region['Full Region Name']
select_name = []


## print chosen brain region
for i in range(len(brain_region_choose)):
    select_index = np.where(brain_region_index == brain_region_choose[i])
    select_name.append(brain_region_name[select_index[0][0]])
    print('select_brain_region:', brain_region_name[select_index[0][0]])


## calculate dff0
for flyname in flynames:   

    ## load atlas
    atlas = io.imread(result_path+ odor_name + flyname+atlaspath+'/Transformed_atlas.tif')
    atlas = atlas[13:38]

    ## select point in each brain region
    select_point = []
    for region in range(len(brain_region_choose)):
        select_point.append([])
        points = np.where(atlas == brain_region_choose[region])
        points_in = []
        for p in range(len(points[0])):
            points_in.append([points[0][p],points[1][p],points[2][p]])
        

        ## random sample some point in voxels
        select_point[region] = points_in


    ## no denoised traces
    tracename = flyname + '-nodenoise-C'
    
    ## trace path
    tracepath = result_path + odor_name + flyname + '/trace/'
                    
    ## calculate dff0
    for odor in range(2,4):
        single_voxel_trace = []
        single_trace = []
        
        for i in range(len(brain_region_choose)):
            single_voxel_trace.append([])
            single_trace.append([])
            for j in range(len(select_point[i])):
                single_voxel_trace[i].append([])
                single_trace[i].append([])

        ## read one file only
        for i in range(file_num):
            trace_right = np.load(tracepath + tracename + str(odor) + '/Trace_%04d.npy'%(file_start + i))
            print("file loaded:" + tracepath + tracename + str(odor) + '/Trace_%04d.npy'%(file_start + i))
            
            for point in range(len(brain_region_choose)): 
                for p in range(len(select_point[point])):
                    single_trace[point][p].append(trace_right[:,select_point[point][p][0],select_point[point][p][1],select_point[point][p][2]])
            if i == 0:
                for point in range(len(select_point)): 
                    trace_voxel = np.zeros((file_frame,1))
                    for p in range(len(select_point[point])):
                        single_voxel_trace[point][p].append(trace_voxel)
                trace_before = trace_right
            else:
                trace = np.concatenate((trace_before,trace_right),0)
                for point in range(len(select_point)): 
                    for p in range(len(select_point[point])):
                        trace_voxel = np.zeros((file_frame,1))
                        for t in range(file_frame):
                            trace_win = np.sort(trace[int(file_frame -frequency*dff0_sw + t):(file_frame + t),select_point[point][p][0],select_point[point][p][1],select_point[point][p][2]])
                            f0 = np.mean(trace_win[0:int(min_percent*frequency*dff0_sw)],0)
                            trace_voxel[t,0] = (trace[file_frame + t,select_point[point][p][0],select_point[point][p][1],select_point[point][p][2]]-f0)/f0
                        single_voxel_trace[point][p].append(trace_voxel)
                trace_before = trace_right
        
        voxel_trace = []
        for i in range(len(select_point)):
            voxel_traces = np.stack(single_voxel_trace[i])
            voxel_traces = np.mean(voxel_traces,0)
            voxel_traces.append(voxel_trace)
        single_voxel_trace = np.stack(voxel_traces)
        single_voxel_trace = single_voxel_trace.reshape(len(select_point),-1)

        single_voxel_trace[np.isnan(single_voxel_trace)] = 0
        single_voxel_trace[np.isinf(single_voxel_trace)] = 0
       
        ## save dff0
        if not os.path.exists(result_path + odor_name +  flyname + '/data/region_voxel_trace'):os.mkdir(result_path + odor_name + flyname + '/data/region_voxel_trace')
        for point in range(len(select_point)): 
            np.save(result_path + odor_name +  flyname + '/data/region_voxel_trace/ave_voxel_dff0_' + str(brain_region_choose[point]) +'_nodenoise_C' + str(odor) + '.npy',single_voxel_trace[point,:])


    ## denoised traces
    tracename = flyname + '-C'

    ## trace path
    tracepath = result_path + odor_name + flyname + '/trace/'
                    
    ## calculate dff0
    for odor in range(2,4):
        single_voxel_trace = []
        single_trace = []
        
        for i in range(len(brain_region_choose)):
            single_voxel_trace.append([])
            single_trace.append([])
            for j in range(len(select_point[i])):
                single_voxel_trace[i].append([])
                single_trace[i].append([])


        ## read one file only
        for i in range(file_num):
            trace_right = np.load(tracepath + tracename + str(odor) + '/Trace_%04d.npy'%(file_start + i))
            print("file loaded:" + tracepath + tracename + str(odor) + '/Trace_%04d.npy'%(file_start + i))
            
            for point in range(len(brain_region_choose)): 
                for p in range(len(select_point[point])):
                    single_trace[point][p].append(trace_right[:,select_point[point][p][0],select_point[point][p][1],select_point[point][p][2]])
            if i == 0:
                for point in range(len(select_point)): 
                    trace_voxel = np.zeros((file_frame,1))
                    for p in range(len(select_point[point])):
                        single_voxel_trace[point][p].append(trace_voxel)
                trace_before = trace_right
            else:
                trace = np.concatenate((trace_before,trace_right),0)
                for point in range(len(select_point)): 
                    for p in range(len(select_point[point])):
                        trace_voxel = np.zeros((file_frame,1))
                        for t in range(file_frame):
                            trace_win = np.sort(trace[int(file_frame -frequency*dff0_sw + t):(file_frame + t),select_point[point][p][0],select_point[point][p][1],select_point[point][p][2]])
                            f0 = np.mean(trace_win[0:int(min_percent*frequency*dff0_sw)],0)
                            trace_voxel[t,0] = (trace[file_frame + t,select_point[point][p][0],select_point[point][p][1],select_point[point][p][2]]-f0)/f0
                        single_voxel_trace[point][p].append(trace_voxel)
                trace_before = trace_right
        
        voxel_trace = []
        for i in range(len(select_point)):
            voxel_traces = np.stack(single_voxel_trace[i])
            voxel_traces = np.mean(voxel_traces,0)
            voxel_traces.append(voxel_trace)
        single_voxel_trace = np.stack(voxel_traces)

        single_voxel_trace[np.isnan(single_voxel_trace)] = 0
        single_voxel_trace[np.isinf(single_voxel_trace)] = 0
       
        ## save dff0
        if not os.path.exists(result_path + odor_name +  flyname + '/data/region_voxel_trace'):os.mkdir(result_path + odor_name + flyname + '/data/region_voxel_trace')
        for point in range(len(select_point)): 
            np.save(result_path + odor_name +  flyname + '/data/region_voxel_trace/ave_voxel_dff0_' + str(brain_region_choose[point]) +'_C' + str(odor) + '.npy',single_voxel_trace[point,:])
