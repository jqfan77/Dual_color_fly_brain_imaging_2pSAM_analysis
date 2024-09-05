##
## average the G7f response map of two indicators
##

import os
from skimage import io
from utils.projection_3d import *

## path
datapath = '../data/'
savepath = '../data/figure/response'
if not os.path.exists(savepath):os.mkdir(savepath)

savepath_Ach = '../data/nsyb-G7f-rAch1h/figure/response'
savepath_5HT = '../data/nsyb-G7f-r5HT1.0/figure/response'


## parameter set
channel_name = ['G7f','Ach']

odor_name = ['OCT','MCH','EA']

cut_z = [28,130] # z-axis cut range

pixel_size = [318.51/512,318.51/512 ,218/218] ## pixel size

trace_std_multi_Ach = io.imread(savepath_Ach + '/response_C2_auc_1.tif')
trace_std_multi_Ach = trace_std_multi_Ach / 65535*10
trace_std_multi_5HT = io.imread(savepath_5HT + '/response_C2_auc_1.tif')
trace_std_multi_5HT = trace_std_multi_5HT / 65535*10
trace_std_multi = (trace_std_multi_Ach + trace_std_multi_5HT)/2


## cut z-axis
trace_std_multi = trace_std_multi[cut_z[0]:cut_z[1],:,:]


## save response map
io.imsave(savepath + '/response_C2_trial_auc_1.tif', (trace_std_multi*65535/10).astype('uint16'))


## plot
projection_3d(trace_std_multi.transpose((1,2,0)),512*pixel_size[0],512*pixel_size[1],trace_std_multi.shape[0]*pixel_size[2],30,None,[0,1.0],0,True,True,os.path.join(savepath, 'response_C2_trial_auc_1.'))


