## 
## This code is used for calculate the statistics of phase and pw for each indicator inselected brain region
##

import numpy as np
import os
import matplotlib.pyplot as plt
import tifffile as tf
import matplotlib as mpl 
import itertools
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests

# parameters
filename = 'phase'
channel_selected = 0

# only left olfactory part
br_index = np.array([64,65,66,55,72,73,74,63])
br_name = np.array(['MBPED_L','MBVL_L','MBML_L','LH_L','SLP_L','SIP_L','SMP_L','CRE_L'])
br_olf = np.array([1,1,1,1,1,1,1,1])
num_region = len(br_index)

# label
stim = np.array([3,1,2,3,2,1,3,1,2,1,2,3,1,2,3,2,3,1,3,2,1,2,3,1,2,3,1,2,1,3,2,3,1,2,3,1,3,1,2,3,1,2,3,2,1,2,1,3,2,1,3,1,2,3,1,2,3,2,1,3,1,2,3,2,1,3,1,3,2,3,2,1,3,2,1,3,1,2,3,2,1,3,1,2,3,2,1,2,3,1,
                 3,1,2,3,1,2,3,2,1,2,3,1,2,1,3,2,1,3,1,3,2,3,1,2,1,2,3,2,3,1,2,3,1,3,2,1,2,3,1,2,1,3,1,2,3,2,3,1,2,1,3,1,3,2,3,1,2,1,2,3,2,1,3,1,2,3,2,3,1,3,1,2,1,3,2,1,3,2,3,1,2,3,2,1,2,1,3,1,2,3])

## ach data
flynames = [
           '20230417-nsyb-G7f-rAch1h/fly2', 
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

# load data_ach
data_path = '../data/'
atlas_path  = '/align_to_atlas'
atlas_z_range = range(13,38)
file_folders = os.listdir(data_path)
list_fly = [1,2,3,4,5,6,7,8,9,10]
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

    data_1[i,0,:] = a[:,0:8]
    data_1[i,1,:] = b[:,0:8]
    print('fly '+str(i)+' done!')

## 5ht data
flynames = [
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
           ]

# load data_5ht
data_path = '../data/'
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

    data_2[i,0,:] = a[:,0:8]
    data_2[i,1,:] = b[:,0:8]
    print('fly '+str(i)+' done!')

## concate G7f, ach and 5ht data
data_2 = data_2*2
data_1 = data_1/30
data_2 = data_2/30

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

## delete none-activity region
if channel_selected == 0:
    data = data_g7f
    dele = [[],[],[],[],[3],[],[],[],[],[],[3,4],[3,4],[],[],[],[0],[],[],[],[0]]
    data_dele = []
    for j in range(data.shape[1]):
        data_dele.append([])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if j not in dele[i]:
                data_dele[j].append(data[i,j])
elif channel_selected == 1:
    data = data_ach
    dele = [[1,4,5],[],[5],[3,4],[1,3,4,5,6],[],[],[4],[],[]]
    dele_region = [3,4]
    data_dele = []
    for j in range(data.shape[1]):
        data_dele.append([])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if j not in dele[i]:
                data_dele[j].append(data[i,j])
    del data_dele[3:5]
    br_name = np.delete(br_name,slice(3,5))
    num_region = len(br_name)
else:
    data = data_5ht 
    dele = [[0,3],[0,1,2,3],[0,3],[3],[],[0,3],[3],[3],[0,3],[]]
    dele_region = [0,3,4]
    data_dele = []
    for j in range(data.shape[1]):
        data_dele.append([])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if j not in dele[i]:
                data_dele[j].append(data[i,j])
    del data_dele[3:5]
    del data_dele[0]
    br_name = np.delete(br_name,slice(3,5))
    br_name = np.delete(br_name,0)
    num_region = len(br_name)

data = data_dele

# Mann-Whitney u test
pairs = list(itertools.combinations(range(num_region), 2))

# add annotation function
def add_stat_annotation(ax, data, pairs):
    p_list = []
    for pair in pairs:
        data1 = data[pair[0]]
        data2 = data[pair[1]]
        stat, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        p_list.append(p)
    
    p_corrected = multipletests(p_list,method = 'fdr_bh')
    p_list = p_corrected[1]
    np.save(figure_save_path + '/' + filename + '_channel' + str(channel_selected) + '_across_region.npy',p_list)
    for index in range(len(pairs)):
        pair = pairs[index]
        p = p_list[index]
        if p < 0.001:
            y_max = max(np.max(data1), np.max(data2))
            x = np.mean(pair)
            ax.plot([pair[0]+1, pair[0]+1, pair[1]+1, pair[1]+1], [y_max*1.1, y_max*1.2, y_max*1.2, y_max*1.1], lw=1.5, color='k')
            ax.text(x+1, y_max*1.2, '***', ha='center', va='bottom', color='k')
        elif p < 0.01:
            y_max = max(np.max(data1), np.max(data2))
            x = np.mean(pair)
            ax.plot([pair[0]+1, pair[0]+1, pair[1]+1, pair[1]+1], [y_max*1.1, y_max*1.2, y_max*1.2, y_max*1.1], lw=1.5, color='k')
            ax.text(x+1, y_max*1.2, '**', ha='center', va='bottom', color='k')
        elif p < 0.05:
            y_max = max(np.max(data1), np.max(data2))
            x = np.mean(pair)
            ax.plot([pair[0]+1, pair[0]+1, pair[1]+1, pair[1]+1], [y_max*1.1, y_max*1.2, y_max*1.2, y_max*1.1], lw=1.5, color='k')
            ax.text(x+1, y_max*1.2, '*', ha='center', va='bottom', color='k')  

# plot box
plt.figure(figsize=(4,1.5))
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.boxplot(data, boxprops=dict(color=color_list[channel_selected]),showmeans=True,meanline=True,meanprops = {'linestyle':'--','color':color_list[channel_selected]},)
add_stat_annotation(ax, data, pairs)
plt.xticks(range(1, num_region + 1), br_name, rotation=90)
for j in range(len(br_name)):
    if br_olf[j] == 1:
        plt.gca().get_xticklabels()[j].set_color('coral')
plt.ylabel(filename + ' (s)')

# save figures
if if_save:
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    plt.savefig(figure_save_path + '/' + filename + '_channel' + str(channel_selected) + '_across_region.pdf', bbox_inches='tight')
    plt.savefig(figure_save_path + '/' + filename + '_channel' + str(channel_selected) + '_across_region.png', bbox_inches='tight')

    