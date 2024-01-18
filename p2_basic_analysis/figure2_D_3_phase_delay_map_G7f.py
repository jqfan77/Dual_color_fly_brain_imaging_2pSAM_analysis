## 
## this code is used for calculate and plot phase map for G7f
## realign should be down in Fiji!
##

import os
from skimage import io
import numpy as np
from utils.projection_3d import *

## path
save_path = '../results/3.odor_random_90_times_OCT_MCH_EA_new/'

data_source = [ '../results/3.odor_random_90_times_OCT_MCH_EA_new/nsyb-G7f-rAch1h/',
               '../results/3.odor_random_90_times_OCT_MCH_EA_new/nsyb-G7f-r5HT1.0/']

flyname_list = [[           
            # '20230417-nsyb-G7f-rAch1h/fly2',
           '20230420-nsyb-G7f-rAch1h/fly2',
           '20230420-nsyb-G7f-rAch1h/fly3',
           '20230428-nsyb-G7f-rAch1h/fly1',
           '20230507-nsyb-G7f-rAch1h/fly1',
           '20230510-nsyb-G7f-rAch1h/fly1',
           '20230510-nsyb-G7f-rAch1h/fly2',
           '20230511-nsyb-G7f-rAch1h/fly2',
           '20230511-nsyb-G7f-rAch1h/fly3',
           '20230515-nsyb-G7f-rAch1h/fly1',
           ],
            [
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
           ]]


## parameter set
cut_z = [14,65] ## cut z-axis range
pixel_size = [318.51/256,318.51/256 ,218/109] ## real pixel size
freq = [[30/13,30/13],[30/13,15/13]] ## frequency
colorscale_phase = [2,7]  ## plot colorbar for phase
colorscale_pw = [3,8] ## plot colorbar for pw
colomap = 'jet'


## plot phase map
if not os.path.exists(save_path + '/figure/phase'):os.mkdir(save_path + '/figure/phase')     
phase = io.imread(data_source[0] + flyname_list[0][0] + '/figure/phase_delay_trial_average/trans_phase_C2.tif')
for j in range(2,3):
    phase_ave = np.zeros_like(phase)
    for type in range(2):
        for fly_index in range(len(flyname_list[type])):
            ## load phase map
            phase = io.imread(data_source[type] + flyname_list[type][fly_index] + '/figure/phase_delay_trial_average/trans_phase_C' + str(j) + '.tif')
            phase[np.isnan(phase)] = 0
            if j == 2:
                phase = phase / freq[type][0]
            if j == 3:
                phase = phase / freq[type][1]
            phase_ave = phase_ave + phase
    phase = phase_ave / (len(flyname_list[0])+len(flyname_list[1]))
    phase_down = phase[cut_z[0]:cut_z[1],:,:]
    projection_3d((phase_down).transpose((1,2,0)),256*pixel_size[0],256*pixel_size[1],phase_down.shape[0]*pixel_size[2],30,colomap,[colorscale_phase[0],colorscale_phase[1]],0,True,True,os.path.join(save_path + 'figure/phase', 'phase_fit_C' + str(j) + '.'))


## plot pw map
if not os.path.exists(save_path + '/figure/pw'):os.mkdir(save_path + '/figure/pw')     
pw = io.imread(data_source[0] + flyname_list[0][0] + '/figure/phase_delay_trial_average/trans_pw_C2.tif')
for j in range(2,3):
    pw_ave = np.zeros_like(pw)
    for type in range(2):
        for fly_index in range(len(flyname_list[type])):
            ## load pw map
            pw = io.imread(data_source[type] + flyname_list[type][fly_index] + '/figure/phase_delay_trial_average/trans_pw_C' + str(j) + '.tif')
            pw[np.isnan(pw)] = 0
            if j == 2:
                pw = pw / freq[type][0]
            if j == 3:
                pw = pw / freq[type][1]
            pw_ave = pw_ave + pw
    pw = pw_ave / (len(flyname_list[0])+len(flyname_list[1]))
    pw_down = pw[cut_z[0]:cut_z[1],:,:]
    projection_3d((pw_down).transpose((1,2,0)),256*pixel_size[0],256*pixel_size[1],pw_down.shape[0]*pixel_size[2],30,colomap ,[colorscale_pw[0],colorscale_pw[1]],0,True,True,os.path.join(save_path + 'figure/pw', 'pw_fit_C' + str(j) + '.'))
