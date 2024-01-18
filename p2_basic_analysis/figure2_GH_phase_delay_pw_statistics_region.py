## 
## This code is used for calculate the statistics of phase and pw for each indicator inselected brain region
##

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import tifffile as tf
import math
import matplotlib as mpl 

# only left olfactory part
br_index = np.array([64,65,66,55,72,73,74,63])
br_name = np.array(['MBPED_L','MBVL_L','MBML_L','LH_L','SLP_L','SIP_L','SMP_L','CRE_L'])
br_olf = np.array([1,1,1,1,1,1,1,1])
num_region = len(br_index)

# label
stim = np.array([3,1,2,3,2,1,3,1,2,1,2,3,1,2,3,2,3,1,3,2,1,2,3,1,2,3,1,2,1,3,2,3,1,2,3,1,3,1,2,3,1,2,3,2,1,2,1,3,2,1,3,1,2,3,1,2,3,2,1,3,1,2,3,2,1,3,1,3,2,3,2,1,3,2,1,3,1,2,3,2,1,3,1,2,3,2,1,2,3,1,
                 3,1,2,3,1,2,3,2,1,2,3,1,2,1,3,2,1,3,1,3,2,3,1,2,1,2,3,2,3,1,2,3,1,3,2,1,2,3,1,2,1,3,1,2,3,2,3,1,2,1,3,1,3,2,3,1,2,1,2,3,2,1,3,1,2,3,2,3,1,3,1,2,1,3,2,1,3,2,3,1,2,3,2,1,2,1,3,1,2,3])
filename = 'pw'
channel_selected = 2


## ach data
flynames = [
        #    'nsyb-G7f-rAch1h/20230417-nsyb-G7f-rAch1h/fly2',
           'nsyb-G7f-rAch1h/20230420-nsyb-G7f-rAch1h/fly2',
           'nsyb-G7f-rAch1h/20230420-nsyb-G7f-rAch1h/fly3', 
           'nsyb-G7f-rAch1h/20230428-nsyb-G7f-rAch1h/fly1', 
           'nsyb-G7f-rAch1h/20230507-nsyb-G7f-rAch1h/fly1', 
           'nsyb-G7f-rAch1h/20230510-nsyb-G7f-rAch1h/fly1',
           'nsyb-G7f-rAch1h/20230510-nsyb-G7f-rAch1h/fly2',
           'nsyb-G7f-rAch1h/20230511-nsyb-G7f-rAch1h/fly2', 
           'nsyb-G7f-rAch1h/20230511-nsyb-G7f-rAch1h/fly3',
           'nsyb-G7f-rAch1h/20230515-nsyb-G7f-rAch1h/fly1',
           ] 

# load data_ach
data_path = '../results/3.odor_random_90_times_OCT_MCH_EA_new/'
atlas_path  = '/align_to_atlas'
atlas_z_range = range(13,38)
file_folders = os.listdir(data_path)
list_fly = [1,2,3,4,5,6,7,8,9]
num_fly = len(list_fly)
num_channel = 2
folder_name = 'figure/phase_delay_trial_average'

# load data
data_1 = np.zeros((num_fly,num_channel,num_region))
for i in range(len(flynames)):
    # load data c2
    the_path = data_path + flynames[i] + '/' + folder_name + '/' + filename + '_map_C2.npy'
    a = np.load(the_path)
    # load data c3
    the_path = data_path + flynames[i] + '/' + folder_name + '/' + filename + '_map_C3.npy'
    b = np.load(the_path)
    # load atlas
    the_path = data_path + flynames[i] + atlas_path + '/' + 'Transformed_atlas_eroded_r5.tif'
    atlas_eroded= tf.imread(the_path)
    atlas_eroded = np.transpose(atlas_eroded,[1,2,0])
    atlas_eroded = atlas_eroded[:,:,atlas_z_range]

    for j in range(num_region):
        mean_a = a[atlas_eroded==br_index[j]]
        if np.size(mean_a,0)==0:
            data_1[i,:,j] = np.nan
            continue
        mean_a = np.nanmean(mean_a,0)
        data_1[i,0,j] = mean_a
        mean_b = b[atlas_eroded==br_index[j]]
        mean_b = np.nanmean(mean_b,0)
        data_1[i,1,j] = mean_b
    print('fly '+str(i)+' done!')

## 5ht data
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

# load data_5ht
data_path = '../results/3.odor_random_90_times_OCT_MCH_EA_new/'
atlas_path  = '/align_to_atlas'
atlas_z_range = range(13,38)
file_folders = os.listdir(data_path)
list_fly = [0,1,2,3,4,5,6,7,8,9]
num_fly = len(list_fly)
num_channel = 2
folder_name = 'figure/phase_delay_trial_average'

# load data
data_2 = np.zeros((num_fly,num_channel,num_region))
for i in range(len(flynames)):
    # load data c2
    the_path = data_path + flynames[i] + '/' + folder_name + '/' + filename + '_map_C2.npy'
    a = np.load(the_path)
    # load data c3
    the_path = data_path + flynames[i] + '/' + folder_name + '/' + filename + '_map_C3.npy'
    b = np.load(the_path)
    # load atlas
    the_path = data_path + flynames[i] + atlas_path + '/' + 'Transformed_atlas_eroded_r5.tif'
    atlas_eroded= tf.imread(the_path)
    atlas_eroded = np.transpose(atlas_eroded,[1,2,0])
    atlas_eroded = atlas_eroded[:,:,atlas_z_range]

    for j in range(num_region):
        mean_a = a[atlas_eroded==br_index[j]]
        if np.size(mean_a,0)==0:
            data_2[i,:,j] = np.nan
            continue
        mean_a = np.nanmean(mean_a,0)
        data_2[i,0,j] = mean_a
        mean_b = b[atlas_eroded==br_index[j]]
        mean_b = np.nanmean(mean_b,0)
        data_2[i,1,j] = mean_b
    print('fly '+str(i)+' done!')


## concate G7f, ach and 5ht data
data_2[:,1,:] = data_2[:,1,:]*2
data_1 = data_1/30*13
data_2 = data_2/30*13

figure_save_path = data_path + 'figure/low_level_statistics_final'
folder = os.path.exists(figure_save_path)
if not folder:
    os.makedirs(figure_save_path)
if_save = True

data_g7f = np.concatenate((np.squeeze(data_1[:,0,:]),np.squeeze(data_2[:,0,:])),axis = 0)
data_ach = np.squeeze(data_1[:,1,:])
data_5ht = np.squeeze(data_2[:,1,:])
print(np.shape(data_g7f))
print(np.shape(data_ach))
print(np.shape(data_5ht))


## plot parameters
color_list = ['#006934','#751C77','#036EB8']
label_list = ['G7f','rAch','r5HT']
if channel_selected == 0:
    data = data_g7f
elif channel_selected == 1:
    data = data_ach
else:
    data = data_5ht 


## plot only average
plt.figure(figsize = (4,2))
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
flag = np.zeros(np.size(data,0))
for i in range(np.size(data,0)):
    if np.sum(np.isnan(np.squeeze(data[i,:])))>0:
        flag[i] = 1
        continue
data = data[flag==0,:]

the_mean = np.nanmean(data,0)
the_std = np.nanstd(data,0)/math.sqrt(np.size(data,0))
plt.errorbar(range(np.size(data,1)),the_mean,yerr=the_std,ecolor=color_list[channel_selected],elinewidth=1,marker='.',mfc=color_list[channel_selected],\
	mec=color_list[channel_selected],mew=1,ms=1,alpha=1,capsize=5,capthick=3,color=color_list[channel_selected], linewidth=2)
plt.xticks(range(num_region),br_name,rotation = 90)
for j in range(len(br_name)):
    if br_olf[j]==1:
        plt.gca().get_xticklabels()[j].set_color('coral') 
plt.ylabel(filename + ' (s)')
plt.ylim((5,15))

if if_save:
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    plt.savefig(figure_save_path + '/' + filename+'_channel'+str(channel_selected)+'_across_region_only_average.pdf',bbox_inches = 'tight')
    plt.savefig(figure_save_path + '/' + filename+'_channel'+str(channel_selected)+'_across_region_only_average.png',bbox_inches = 'tight')


## plot across region
plt.figure(figsize = (4,1.5))
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
flag = np.zeros(np.size(data,0))
for i in range(np.size(data,0)):
    if np.sum(np.isnan(np.squeeze(data[i,:])))>0:
        flag[i] = 1
        continue
    plt.plot(np.squeeze(data[i,:]),color = color_list[channel_selected],alpha = 0.2)
data = data[flag==0,:]
the_mean = np.nanmean(data,0)
the_std = np.nanstd(data,0)/math.sqrt(np.size(data,0))
plt.errorbar(range(np.size(data,1)),the_mean,yerr=the_std,ecolor=color_list[channel_selected],elinewidth=1,marker='.',mfc=color_list[channel_selected],\
	mec=color_list[channel_selected],mew=1,ms=1,alpha=1,capsize=5,capthick=3,color=color_list[channel_selected], linewidth=2)
plt.xticks(range(num_region),br_name,rotation = 90)
for j in range(len(br_name)):
    if br_olf[j]==1:
        plt.gca().get_xticklabels()[j].set_color('coral') 
plt.ylabel(filename + ' (s)')
plt.ylim((5,15))

if if_save:
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    plt.savefig(figure_save_path + '/' + filename+'_channel'+str(channel_selected)+'_across_region.pdf',bbox_inches = 'tight')
    plt.savefig(figure_save_path + '/' + filename+'_channel'+str(channel_selected)+'_across_region.png',bbox_inches = 'tight')