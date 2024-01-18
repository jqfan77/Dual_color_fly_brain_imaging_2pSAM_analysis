import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
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
filename = 'phase'

# ach data
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
    print('fly '+str(id)+' done!')

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
    print('fly '+str(id)+' done!')

## concate ach and 5ht data
data_2[:,1,:] = data_2[:,1,:]*2
data_1 = data_1/30*13
data_2 = data_2/30*13

figure_save_path = data_path + 'figure/low_level_statistics_final'
folder = os.path.exists(figure_save_path)
if not folder:
    os.makedirs(figure_save_path)
if_save = True

data_g7f_origin = np.concatenate((np.squeeze(data_1[:,0,:]),np.squeeze(data_2[:,0,:])),axis = 0)
data_ach_origin = np.squeeze(data_1[:,1,:])
data_5ht_origin = np.squeeze(data_2[:,1,:])

## plot each region
plt.figure(figsize=(8,4))
grid = plt.GridSpec(2,4, wspace=0.3, hspace=0.5)
for kkk in range(num_region):
    data_g7f = np.squeeze(data_g7f_origin[:,kkk])
    data_ach = np.squeeze(data_ach_origin[:,kkk])
    data_5ht = np.squeeze(data_5ht_origin[:,kkk])
    color_list = ['#006934','#751C77','#036EB8']
    label_list = ['G7f','rAch','r5HT']
    ax = plt.subplot(grid[int(kkk/4),kkk%4])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for i in range(3):
        if i == 0:
            a = data_g7f
        elif i == 1:
            a = data_ach
        else:
            a = data_5ht
        num_fly = np.size(a,0)
        the_mean = np.nanmean(a)
        the_std = np.nanstd(a,ddof = 1)/math.sqrt(num_fly-np.sum(np.isnan(a)))
        plt.bar([i], the_mean, width=0.7,yerr = the_std,error_kw = {'ecolor' : '0.2', 'capsize' :3 },
                            alpha=0.7,facecolor = 'white',edgecolor=color_list[i],linewidth=1.5,label = label_list[i])
    plt.xticks(range(3),label_list)
    plt.ylabel(filename+' (s)')
    y = 3

    # * g7f and ach
    res = mannwhitneyu(data_g7f,data_ach)
    p = res.pvalue
    if p<0.05 and p>=0.01:
        plt.text(0.5,y,'*',verticalalignment = 'center', horizontalalignment = 'center')
    elif p<0.01 and p>0.001:
        plt.text(0.5,y,'**',verticalalignment = 'center', horizontalalignment = 'center')
    elif p<0.001 and p>=0.0001:
        plt.text(0.5,y,'***',verticalalignment = 'center', horizontalalignment = 'center') 
    elif p<0.0001:
        plt.text(0.5,y,'****',verticalalignment = 'center', horizontalalignment = 'center') 

    # * g7f and 5ht
    res = mannwhitneyu(data_g7f,data_5ht)
    p = res.pvalue
    if p<0.05 and p>=0.01:
        plt.text(1,y,'*',verticalalignment = 'center', horizontalalignment = 'center')
    elif p<0.01 and p>0.001:
        plt.text(1,y,'**',verticalalignment = 'center', horizontalalignment = 'center')
    elif p<0.001 and p>=0.0001:
        plt.text(1,y,'***',verticalalignment = 'center', horizontalalignment = 'center') 
    elif p<0.0001:
        plt.text(1,y,'****',verticalalignment = 'center', horizontalalignment = 'center') 

    # * ach and 5ht
    res = mannwhitneyu(data_ach,data_5ht)
    p = res.pvalue
    if p<0.05 and p>=0.01:
        plt.text(1.5,y,'*',verticalalignment = 'center', horizontalalignment = 'center')
    elif p<0.01 and p>0.001:
        plt.text(1.5,y,'**',verticalalignment = 'center', horizontalalignment = 'center')
    elif p<0.001 and p>=0.0001:
        plt.text(1.5,y,'***',verticalalignment = 'center', horizontalalignment = 'center') 
    elif p<0.0001:
        plt.text(1.5,y,'****',verticalalignment = 'center', horizontalalignment = 'center') 
    plt.title(br_name[kkk])
    # plt.ylim(0,5)

if if_save:
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    plt.savefig(figure_save_path + '/' + filename+'_each_region.pdf',bbox_inches = 'tight')
    plt.savefig(figure_save_path + '/' + filename+'_each_region.png',bbox_inches = 'tight')
