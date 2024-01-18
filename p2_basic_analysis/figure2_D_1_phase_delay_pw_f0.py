## 
## This code is used for calculate the phase map of each fly
## the start of response was defined by the first time that the reponse reach the average of 20 points before the stim window
##

import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import io

## only left part
br_index = [64,65,66,55,72,73,74,63,84,59,4,23,26,56,79,75,76,77,60,80,82,85,67]
br_name = ['MBPED_L','MBVL_L','MBML_L','LH_L','SLP_L','SIP_L','SMP_L','CRE_L','SCL_L','ICL_L',
'NO','EB','FB','LAL_L','AOTU_L','AVLP_L','PVLP_L','IVLP_L','VES_L','GOR_L','SPS_L','EPA_L','FLA_L']
br_olf = np.array([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
num_region = len(br_index)


## data
# flynames = [
#            'nsyb-G7f-rAch1h/20230417-nsyb-G7f-rAch1h/fly2', 
#            'nsyb-G7f-rAch1h/20230420-nsyb-G7f-rAch1h/fly2',
#            'nsyb-G7f-rAch1h/20230420-nsyb-G7f-rAch1h/fly3',
#            'nsyb-G7f-rAch1h/20230428-nsyb-G7f-rAch1h/fly1',
#            'nsyb-G7f-rAch1h/20230507-nsyb-G7f-rAch1h/fly1', 
#            'nsyb-G7f-rAch1h/20230510-nsyb-G7f-rAch1h/fly1',
#            'nsyb-G7f-rAch1h/20230510-nsyb-G7f-rAch1h/fly2',
#            'nsyb-G7f-rAch1h/20230511-nsyb-G7f-rAch1h/fly2', 
#            'nsyb-G7f-rAch1h/20230511-nsyb-G7f-rAch1h/fly3',
#            'nsyb-G7f-rAch1h/20230515-nsyb-G7f-rAch1h/fly1',
#            ] 

flynames = [
           'nsyb-G7f-r5HT1.0/20230429-nsyb-G7f-r5HT1.0/fly1',
           'nsyb-G7f-r5HT1.0/20230506-nsyb-G7f-r5HT1.0/fly1',
           'nsyb-G7f-r5HT1.0/20230513-nsyb-G7f-r5HT1.0/fly1',
           'nsyb-G7f-r5HT1.0/20230513-nsyb-G7f-r5HT1.0/fly2',
           'nsyb-G7f-r5HT1.0/20230516-nsyb-G7f-r5HT1.0/fly2',
           'nsyb-G7f-r5HT1.0/20230516-nsyb-G7f-r5HT1.0/fly4',
           'nsyb-G7f-r5HT1.0/20230517-nsyb-G7f-r5HT1.0/fly1',
           'nsyb-G7f-r5HT1.0/20230601-nsyb-G7f-r5HT1.0/fly1',
           'nsyb-G7f-r5HT1.0/20230601-nsyb-G7f-r5HT1.0/fly3',
           'nsyb-G7f-r5HT1.0/20230603-nsyb-G7f-r5HT1.0/fly1',
           ]


## path
datapath = '../results/3.odor_random_90_times_OCT_MCH_EA_new/'

atlaspath = '/align_to_atlas'

result_path = '../results/3.odor_random_90_times_OCT_MCH_EA_new/'

num_fly = len(flynames)

atlas_z_range = range(13,38)

channel_selected = 'C3'


def compute_phase(data,atlas,atlas_eroded,br_index):
    # compute phase map
    num_tp = np.size(data,4)
    num_z = np.size(data,2)
    num_x = np.size(data,0)
    num_y = np.size(data,1)
    phase_map = np.zeros((num_x,num_y,num_z))
    near = np.zeros((num_x,num_y,num_z,num_tp))
    trunc = np.zeros((num_x,num_y,num_z)) # trunc for find f0

    for i in range(num_x):
        for j in range(num_y):
            for k in range(num_z):
                if atlas[i,j,k]==0:
                    continue
                a = np.squeeze(data[i,j,k,:,:])
                a = np.mean(a,0)
                x = np.linspace(1,len(a),len(a))
                z1 = np.polyfit(x,a,15)
                p1 = np.poly1d(z1)
                near[i,j,k,:] = p1(x)
                trunc[i,j,k] = np.mean(a[0:int(len(a)/2)])

    for i in range(num_x):
        for j in range(num_y):
            for k in range(num_z):
                if atlas[i,j,k]==0:
                    continue
                a = near[i,j,k,:]
                max_index =  np.argmax(a)
                min_index = 0
                for mm in range(max_index-1,-1,-1):
                    if a[mm] <= trunc[i,j,k]:
                        min_index = mm
                        break
                phase_map[i,j,k] = max_index - min_index
    print('phase map done!')
    phase_show = phase_map
    plt.imshow(phase_show.max(2))
    plt.colorbar()

    # compute pulse width map
    pw_map = np.zeros((num_x,num_y,num_z))
    for i in range(num_x):
        for j in range(num_y):
            for k in range(num_z):
                if atlas[i,j,k]==0:
                    continue
                a = near[i,j,k,:]
                the_max = np.max(a)
                the_thresh = (the_max+trunc[i,j,k])/2

                # ind_1
                inds = np.array(np.where(a>=the_thresh))
                ind_1 = inds[inds<np.argmax(a)]
                if np.size(ind_1)==0:# if no ind_1, then the start is the largest
                    pw_map[i,j,k] = 0
                    continue
                if np.size(ind_1)>1:
                    ind_1 = ind_1[0]
                # ind_2
                inds = np.array(np.where(a<=the_thresh))
                ind_2 = inds[inds>np.argmax(a)]
                if np.size(ind_2)>0:
                    if np.size(ind_2)>1:
                        ind_2 = ind_2[0]
                else:
                    ind_2 = np.size(a)-1
                pw_map[i,j,k] = ind_2-ind_1
    print('pw map done!')
    pw_show = pw_map
    plt.imshow(pw_show.max(2))
    plt.colorbar()

    return phase_map,pw_map


for flyname in flynames:
    the_save_path = datapath + flyname + '/figure/phase_delay_trial_average'

    # load atlas
    atlas = io.imread(datapath + flyname + '/align_to_atlas/Transformed_atlas.tif')
    atlas = np.transpose(atlas,[1,2,0])
    atlas = atlas[:,:,atlas_z_range]

    # load atlas_eroded
    atlas_eroded = io.imread(datapath + flyname + '/align_to_atlas/Transformed_atlas.tif')
    atlas_eroded = np.transpose(atlas_eroded,[1,2,0])
    atlas_eroded = atlas_eroded[:,:,atlas_z_range]

    # load dff0_c2
    if channel_selected =='C2':
        the_path = datapath + flyname + '/data/dff0_-20-20_down1_'+channel_selected+'.npy'
    if channel_selected =='C3':
        if 'Ach' in flyname:
            the_path = datapath + flyname + '/data/dff0_-20-20_down1_'+channel_selected+'.npy'
        if '5HT' in flyname:
            the_path = datapath + flyname + '/data/dff0_-40-40_down2_'+channel_selected+'.npy'
    a = np.load(the_path)
    a = np.transpose(a,[3,4,2,0,1])
    print(np.shape(a))
    
    # compute phase and save
    [phase_map,pw_map] = compute_phase(a,atlas,atlas_eroded,br_index)
    folder = os.path.exists(the_save_path)
    if not folder:
        os.makedirs(the_save_path)
    np.save(the_save_path + '/'+'phase_map_'+channel_selected+'.npy',phase_map)
    np.save(the_save_path + '/'+'pw_map_'+channel_selected+'.npy',pw_map)
    print('fly '+ str(id) + ' done!')
    