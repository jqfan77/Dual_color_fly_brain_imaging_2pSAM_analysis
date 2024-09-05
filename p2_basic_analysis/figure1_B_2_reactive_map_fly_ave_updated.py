##
## calculate the mean of each fly's std response
## this code should be run after realign the map in Fiji
## plot the 3d image of response
##

import numpy as np
import os
from skimage import io
from utils.projection_3d import *

## data
flynames = [
        #    '20230417-nsyb-G7f-rAch1h/fly2', 
           '20230420-nsyb-G7f-rAch1h/fly2',
        #    '20230420-nsyb-G7f-rAch1h/fly3',
        #    '20230428-nsyb-G7f-rAch1h/fly1',
        #    '20230507-nsyb-G7f-rAch1h/fly1', 
           '20230510-nsyb-G7f-rAch1h/fly1',
           '20230510-nsyb-G7f-rAch1h/fly2',
        #    '20230511-nsyb-G7f-rAch1h/fly2',
           '20230511-nsyb-G7f-rAch1h/fly3',
           '20230515-nsyb-G7f-rAch1h/fly1'
           ] 

# flynames = [
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


## path
datapath = '../data/'
savepath = '../data/nsyb-G7f-rAch1h/figure/response'
# savepath = '../results/3.odor_random_90_times_OCT_MCH_EA_new/nsyb-G7f-r5HT1.0/figure/response'
if not os.path.exists(savepath):
    os.mkdir(savepath)


## parameter set
channel_name = ['G7f','Ach']
odor_name = ['OCT','MCH','EA']
cut_z = [28,130] # z-axis cut range
print(cut_z)
pixel_size = [318.51/512,318.51/512 ,218/218] ## pixel size


## figures
for c in range(2,4):

    # ## calculate response map average in flys
    # trace_std_multi = []
    # for flyname in flynames:
    #     trace_std = io.imread(datapath + flyname + '/figure/raw' + '/Transformed_response_C' + str(c) + '_std_1.tif')
    #     trace_std = trace_std / 65535
    #     trace_std_multi.append(trace_std)
    # trace_std_multi = np.mean(np.stack(trace_std_multi),0)
    # ## save average response map
    # io.imsave(savepath + '/response_C' + str(c) + '_std_1.tif', (trace_std_multi*65535).astype('uint16'))
    # ## cut z-axis
    # trace_std_multi = trace_std_multi[cut_z[0]:cut_z[1],:,:]
    # ## plot 3d image 
    # if c == 2:
    #     projection_3d(trace_std_multi.transpose((1,2,0)),512*pixel_size[0],512*pixel_size[1],trace_std_multi.shape[0]*pixel_size[2],30,None,[0,0.06],0,True,True,os.path.join(savepath, 'response_C'+str(c) +'_std_1.'))      
    # if c == 3:
    #     projection_3d(trace_std_multi.transpose((1,2,0)),512*pixel_size[0],512*pixel_size[1],trace_std_multi.shape[0]*pixel_size[2],30,None,[0,0.04],0,True,True,os.path.join(savepath, 'response_C'+str(c) +'_std_1.'))


    # ## calculate response map average in flys
    # trace_std_multi = []
    # for flyname in flynames:
    #     trace_std = io.imread(datapath + flyname + '/figure/raw' + '/Transformed_response_C' + str(c) + '_sem.tif')
    #     trace_std = trace_std / 65535
    #     trace_std_multi.append(trace_std)
    # trace_std_multi = np.mean(np.stack(trace_std_multi),0)
    # ## save average response map
    # io.imsave(savepath + '/response_C' + str(c) + '_sem.tif', (trace_std_multi*65535).astype('uint16'))
    # ## cut z-axis
    # trace_std_multi = trace_std_multi[cut_z[0]:cut_z[1],:,:]
    # ## plot 3d image 
    # if c == 2:
    #     projection_3d(trace_std_multi.transpose((1,2,0)),512*pixel_size[0],512*pixel_size[1],trace_std_multi.shape[0]*pixel_size[2],30,None,[0,0.012],0,True,True,os.path.join(savepath, 'response_C'+str(c) +'_sem.'))      
    # if c == 3:
    #     projection_3d(trace_std_multi.transpose((1,2,0)),512*pixel_size[0],512*pixel_size[1],trace_std_multi.shape[0]*pixel_size[2],30,None,[0,0.008],0,True,True,os.path.join(savepath, 'response_C'+str(c) +'_sem.'))


    ## calculate response map average in flys
    trace_std_multi = []
    for flyname in flynames:
        trace_std = io.imread(datapath + flyname + '/figure/raw' + '/Transformed_response_C' + str(c) + '_auc_1.tif')
        trace_std = trace_std / 65535 * 10
        trace_std_multi.append(trace_std)
    trace_std_multi = np.mean(np.stack(trace_std_multi),0)
    ## save average response map
    io.imsave(savepath + '/response_C' + str(c) + '_auc_1.tif', (trace_std_multi/10*65535).astype('uint16'))
    ## cut z-axis
    trace_std_multi = trace_std_multi[cut_z[0]:cut_z[1],:,:]
    ## plot 3d image 
    if c == 2:
        projection_3d(trace_std_multi.transpose((1,2,0)),512*pixel_size[0],512*pixel_size[1],trace_std_multi.shape[0]*pixel_size[2],30,None,[0,0.9],0,True,True,os.path.join(savepath, 'response_C'+str(c) +'_auc_1.'))      
    if c == 3:
        projection_3d(trace_std_multi.transpose((1,2,0)),512*pixel_size[0],512*pixel_size[1],trace_std_multi.shape[0]*pixel_size[2],30,None,[0,0.9],0,True,True,os.path.join(savepath, 'response_C'+str(c) +'_auc_1.'))


    # ## calculate response map average in flys
    # trace_std_multi = []
    # for flyname in flynames:
    #     trace_std = io.imread(datapath + flyname + '/figure/raw' + '/Transformed_response_C' + str(c) + '_avg.tif')
    #     trace_std = trace_std / 65535
    #     trace_std_multi.append(trace_std)
    # trace_std_multi = np.mean(np.stack(trace_std_multi),0)
    # ## save average response map
    # io.imsave(savepath + '/response_C' + str(c) + '_avg.tif', (trace_std_multi*65535).astype('uint16'))
    # ## cut z-axis
    # trace_std_multi = trace_std_multi[cut_z[0]:cut_z[1],:,:]
    # ## plot 3d image 
    # if c == 2:
    #     projection_3d(trace_std_multi.transpose((1,2,0)),512*pixel_size[0],512*pixel_size[1],trace_std_multi.shape[0]*pixel_size[2],30,None,[0,0.15],0,True,True,os.path.join(savepath, 'response_C'+str(c) +'_avg.'))      
    # if c == 3:
    #     projection_3d(trace_std_multi.transpose((1,2,0)),512*pixel_size[0],512*pixel_size[1],trace_std_multi.shape[0]*pixel_size[2],30,None,[0,0.1],0,True,True,os.path.join(savepath, 'response_C'+str(c) +'_avg.'))
