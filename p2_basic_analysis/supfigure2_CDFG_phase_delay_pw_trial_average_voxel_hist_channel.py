import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import tifffile as tf

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
if_save = True
the_color = ['#751C77','#036EB8']


## data
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

figure_save_path = data_path + 'figure/low_level_statistics_final'
folder = os.path.exists(figure_save_path)
if not folder:
    os.makedirs(figure_save_path)
atlas_path  = '/align_to_atlas'
atlas_z_range = range(13,38)
file_folders = os.listdir(data_path)
list_fly = [1,2,3,4,5,6,7,8,9]
num_fly = len(list_fly)
num_channel = 2
folder_name = 'figure/phase_delay_trial_average'
# load data
plt.figure(figsize=(8,4))
grid = plt.GridSpec(2,4, wspace=0.3, hspace=0.5)
for kkk in range(num_region):
    flag = []
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
        mean_a = a[atlas_eroded==br_index[kkk]]
        if np.size(mean_a,0)==0:
            continue
        mean_b = b[atlas_eroded==br_index[kkk]]
        mean_a = mean_a/30*13
        mean_b = mean_b/30*13
        flag.extend(mean_b-mean_a)
        # print('fly '+str(id)+' done!')
    plt.subplot(grid[int(kkk/4),kkk%4])
    weights = np.ones_like(flag)/float(np.size(flag))
    plt.hist(flag,color = the_color[0],bins = 20,alpha = 0.7,range=(-5,5),weights = weights)
    plt.ylim((0,0.4))
    plt.title(br_name[kkk])
if if_save:
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    plt.savefig(figure_save_path + '/' + filename+'_voxel_compare_channel_ach.pdf',bbox_inches = 'tight')
    plt.savefig(figure_save_path + '/' + filename+'_voxel_compare_channel_ach.png',bbox_inches = 'tight')
# plt.show()




flynames = [
           'nsyb-G7f-r5HT1.0/20230429-nsyb-G7f-r5HT1.0/fly1',
           'nsyb-G7f-r5HT1.0/20230506-nsyb-G7f-r5HT1.0/fly1',
           'nsyb-G7f-r5HT1.0/20230513-nsyb-G7f-r5HT1.0/fly1',
           'nsyb-G7f-r5HT1.0/20230513-nsyb-G7f-r5HT1.0/fly2',
           'nsyb-G7f-r5HT1.0/20230516-nsyb-G7f-r5HT1.0/fly2',
           'nsyb-G7f-r5HT1.0/20230516-nsyb-G7f-r5HT1.0/fly4',
           'nsyb-G7f-r5HT1.0/20230517-nsyb-G7f-r5HT1.0/fly1',
           'nsyb-G7f-r5HT1.0/20230601-nsyb-G7f-r5HT1.0/fly1', ## 缺LH
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
plt.figure(figsize=(8,4))
grid = plt.GridSpec(2,4, wspace=0.3, hspace=0.5)
for kkk in range(num_region):
    flag = []
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
        mean_a = a[atlas_eroded==br_index[kkk]]
        if np.size(mean_a,0)==0:
            continue
        mean_b = b[atlas_eroded==br_index[kkk]]
        mean_a = mean_a/30*13
        mean_b = mean_b/30*2*13
        flag.extend(mean_b-mean_a)
        # print('fly '+str(id)+' done!')
    plt.subplot(grid[int(kkk/4),kkk%4])
    weights = np.ones_like(flag)/float(np.size(flag))
    plt.hist(flag,color = the_color[1],alpha = 0.7,bins = 20,range=(-15,15),weights = weights)
    plt.ylim((0,0.4))
    plt.title(br_name[kkk])
if if_save:
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    plt.savefig(figure_save_path + '/' + filename+'_voxel_compare_channel_5ht.pdf',bbox_inches = 'tight')
    plt.savefig(figure_save_path + '/' + filename+'_voxel_compare_channel_5ht.png',bbox_inches = 'tight')
# plt.show()


