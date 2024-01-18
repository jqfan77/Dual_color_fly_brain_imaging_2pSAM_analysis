## 
## this code is used for calculate and plot phase map for each indicator
## realign should be down in Fiji!
##

import os
from skimage import io
import numpy as np
from utils.projection_3d import *

## realign should be down in Fiji!
realign = False


## path
atlaspath = '/align_to_atlas'

result_path = '../results/3.odor_random_90_times_OCT_MCH_EA_new/'

data_source =  '../results/3.odor_random_90_times_OCT_MCH_EA_new/nsyb-G7f-rAch1h/'
# data_source =  ../../results/3.odor_random_90_times_OCT_MCH_EA_new/nsyb-G7f-r5HT1.0/'

flyname_list = [
        #    '20230417-nsyb-G7f-rAch1h/fly2',
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

# flyname_list = [
#            '20230429-nsyb-G7f-r5HT1.0/fly1',
#            '20230506-nsyb-G7f-r5HT1.0/fly1',
#            '20230513-nsyb-G7f-r5HT1.0/fly1',
#            '20230513-nsyb-G7f-r5HT1.0/fly2',
#            '20230516-nsyb-G7f-r5HT1.0/fly2',
#            '20230516-nsyb-G7f-r5HT1.0/fly4',
#            '20230517-nsyb-G7f-r5HT1.0/fly1',
#            '20230601-nsyb-G7f-r5HT1.0/fly1',
#            '20230601-nsyb-G7f-r5HT1.0/fly3',
#            '20230603-nsyb-G7f-r5HT1.0/fly1',
#            ]


## parameter set
channel_name = ['G7f','5HT']

odor_name = ['OCT','MCH','EA']

if 'Ach' in data_source:
    freq = [30/13,30/13]  # frequency
    colorscale_phase = [8,7] # plot colorbar for phase
    colorscale_pw = [10,8] # plot colorbar for phase width
if '5HT' in data_source:
    freq = [30/13,15/13]
    colorscale_phase = [8,11]
    colorscale_pw = [10,18]

cut_z = [14,65]
pixel_size = [318.51/256,318.51/256 ,218/109]

fly_num = len(flyname_list)

stim_random_list_1 = [3,1,2,3,2,1,3,1,2,1,2,3,1,2,3,2,3,1]
stim_random_list = np.array(stim_random_list_1)

if realign == False:
    ## cal phase map
    for fly_index in range(len(flyname_list)):
        ## load atlas
        fly_name = flyname_list[fly_index]
        
        ## cal phase
        for c in range(2,4):
            phase = np.load(data_source + fly_name + '/figure/phase_delay_trial_average/phase_map_C'+str(c)+'.npy')
            phase[np.isinf(phase)] = np.nan
            if not os.path.exists(data_source + fly_name  + '/figure/raw'):os.mkdir(data_source + fly_name  + '/figure/raw')     
            io.imsave(data_source + fly_name + '/figure/phase_delay_trial_average/phase_C'+str(c)+ '.tif', (phase.transpose((2,0,1))).astype('uint16'))

        ## cal pw
        for c in range(2,4):
            pw = np.load(data_source + fly_name + '/figure/phase_delay_trial_average/pw_map_C'+str(c)+'.npy')
            pw[np.isinf(pw)] = np.nan
            if not os.path.exists(data_source + fly_name  + '/figure/raw'):os.mkdir(data_source + fly_name  + '/figure/raw')     
            io.imsave(data_source + fly_name + '/figure/phase_delay_trial_average/pw_C'+str(c)+ '.tif', (pw.transpose((2,0,1))).astype('uint16'))

if realign == True:
    #plot phase map for each fly
    for fly_index in range(len(flyname_list)):

        fly_name = flyname_list[fly_index]

        ## plot phase map
        if not os.path.exists(data_source + fly_name + '/figure/phase'):os.mkdir(data_source + fly_name + '/figure/phase')
        for i in range(2,4):
            phase = io.imread(data_source + fly_name + '/figure/phase_delay_trial_average/trans_phase_C' + str(i) + '.tif')
            if i == 2:
                phase = phase / freq[0]
            if i == 3:
                phase = phase / freq[1]
            phase_down = phase[cut_z[0]:cut_z[1],:,:]
            if i == 2:
                colorscale = colorscale_phase[0]
            if i == 3:
                colorscale = colorscale_phase[1]
            projection_3d(phase_down.transpose((1,2,0)),256*pixel_size[0],256*pixel_size[1],phase_down.shape[0]*pixel_size[2],30,None,[0,colorscale],0,True,True,os.path.join(data_source + fly_name + '/figure/phase', 'trans_phase_fit_C' + str(i) + '.'))
                
        ## load pw map
        if not os.path.exists(data_source + fly_name + '/figure/pw'):os.mkdir(data_source + fly_name + '/figure/pw')
        for i in range(2,4):
            pw = io.imread(data_source + fly_name + '/figure/phase_delay_trial_average/trans_pw_C' + str(i) + '.tif')
            if i == 2:
                pw = pw / freq[0]
            if i == 3:
                pw = pw / freq[1]
            pw_down = pw[cut_z[0]:cut_z[1],:,:]
            if i == 2:
                colorscale = colorscale_pw[0]
            if i == 3:
                colorscale = colorscale_pw[1]
            projection_3d(pw_down.transpose((1,2,0)),256*pixel_size[0],256*pixel_size[1],pw_down.shape[0]*pixel_size[2],30,None,[0,colorscale],0,True,True,os.path.join(data_source + fly_name + '/figure/pw', 'trans_pw_fit_C' + str(i) + '.'))
                
                
    ## plot phase map for fly average
    ## load a phase map
    fly_name = flyname_list[0]
    phase = io.imread(data_source + fly_name + '/figure/phase_delay_trial_average/trans_phase_C2.tif')

    ## plot phase map
    if not os.path.exists(data_source + '/figure/phase'):os.mkdir(data_source + '/figure/phase') 
    for i in range(2,4):
        phase_ave = np.zeros_like(phase)
        for fly_index in range(len(flyname_list)):
            fly_name = flyname_list[fly_index]
            phase = io.imread(data_source + fly_name + '/figure/phase_delay_trial_average/trans_phase_C' + str(i) + '.tif')
            phase[np.isnan(phase)] = 0
            if i == 2:
                phase = phase / freq[0]
            if i == 3:
                phase = phase / freq[1]
            phase_ave = phase_ave + phase
        
        ## fly average
        phase = phase_ave / len(flyname_list)
        phase_down = phase[cut_z[0]:cut_z[1],:,:]
        if i == 2:
            colorscale = colorscale_phase[0]
        if i == 3:
            colorscale = colorscale_phase[1]
        projection_3d(phase_down.transpose((1,2,0)),256*pixel_size[0],256*pixel_size[1],phase_down.shape[0]*pixel_size[2],30,'jet',[2,colorscale],0,True,True,os.path.join(data_source + '/figure/phase', 'phase_fit_C' + str(i) + '.'))
     
    ## plot pw map
    if not os.path.exists(data_source + '/figure/pw'):os.mkdir(data_source + '/figure/pw')     
    for i in range(2,4):
        pw_ave = np.zeros_like(pw)
        for fly_index in range(len(flyname_list)):
            fly_name = flyname_list[fly_index]
            pw = io.imread(data_source + fly_name + '/figure/phase_delay_trial_average/trans_pw_C' + str(i) + '.tif')
            pw[np.isnan(pw)] = 0
            if i == 2:
                pw = pw / freq[0]
            if i == 3:
                pw = pw / freq[1]
            pw_ave = pw_ave + pw
        pw = pw_ave / len(flyname_list)
        pw_down = pw[cut_z[0]:cut_z[1],:,:]
        if i == 2:
            colorscale = colorscale_pw[0]
        if i == 3:
            colorscale = colorscale_pw[1]
        projection_3d(pw_down.transpose((1,2,0)),256*pixel_size[0],256*pixel_size[1],pw_down.shape[0]*pixel_size[2],30,'jet',[3,colorscale],0,True,True,os.path.join(data_source + '/figure/pw', 'pw_fit_C' + str(i) + '.'))
      
