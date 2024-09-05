import numpy as np
from utils.p2_pca_lda import *
import tifffile as tf
import random
# label
stim = np.array([3,1,2,3,2,1,3,1,2,1,2,3,1,2,3,2,3,1,3,2,1,2,3,1,2,3,1,2,1,3,2,3,1,2,3,1,3,1,2,3,1,2,3,2,1,2,1,3,2,1,3,1,2,3,1,2,3,2,1,3,1,2,3,2,1,3,1,3,2,3,2,1,3,2,1,3,1,2,3,2,1,3,1,2,3,2,1,2,3,1,
                 3,1,2,3,1,2,3,2,1,2,3,1,2,1,3,2,1,3,1,3,2,3,1,2,1,2,3,2,3,1,2,3,1,3,2,1,2,3,1,2,1,3,1,2,3,2,3,1,2,1,3,1,3,2,3,1,2,1,2,3,2,1,3,1,2,3,2,3,1,3,1,2,1,3,2,1,3,2,3,1,2,3,2,1,2,1,3,1,2,3])
# brain region
br_index_1 = np.array([17,18,19,64,65,66])
br_index = np.array([64])
br_name = np.array(['MBPED_L'])

#######################
file_folder_list_1 = ['20240802-OK107-G7f-rACh/fly3/data','20240804-OK107-G7f-rACh/fly1/data','20240804-OK107-G7f-rACh/fly2/data',
                      '20240805-OK107-G7f-rACh/fly1/data','20240806-OK107-G7f-rACh/fly1/data','20240811-OK107-G7f-rACh/fly2/data']
file_folder_list_2 = ['20240802-OK107-fly3','20240804-OK107-fly1','20240804-OK107-fly2',
                      '20240805-OK107-fly1','20240806-OK107-fly1','20240811-OK107-fly2']
#############################
## read data
data_path = '../../complementary_data/MB-gal4'
atlas_path = data_path
result_path = '../results/OK107-G7f-rACh'
file_name = ['dff0_-6-80_down2_all_C2_one_f0','dff0_-6-80_down2_all_C3_one_f0']
atlas_name = 'align_to_atlas/Transformed_atlas.tif'
atlas_eroded_name = 'align_to_atlas/Transformed_atlas_eroded_r5.tif'
num_fly = len(file_folder_list_1)
atlas_z_range = range(13,38)
num_channel = 2

dff0_thresh = 0.5
# cv fold
cv_fold = 5
# channel
list_channel_selected = [0,1,2]
num_channel_selected = len(list_channel_selected)
# list_odor_choice
list_odor_choice = [0]
num_odor_choice = len(list_odor_choice)
# thresh_pca_exp_var_2
list_thresh_pca_exp_var_2 = np.arange(0,1,0.02)
num_thresh = len(list_thresh_pca_exp_var_2)
#######
fly_selected = 1
print(file_folder_list_1[fly_selected])

## plot
if_plot_figure = False
if_save_figure = True

# if_shuffle
if_shuffle = [False,True]
if_shuffle_flag = ['No_Shuffle','Shuffle']
num_shuffle_choice = len(if_shuffle)

## parameters for PCA
pca_tp_range = range(4,18)
## parameters for LDA and SVM
lda_tp_selected = 6
svm_tp_range = range(4,18)


###################################
# load data
file_folder_1 = file_folder_list_1[fly_selected]
path_file_0 = data_path + '/' + file_folder_1 + '/' + file_name[0] + '.npy'
path_file_1 = data_path + '/' + file_folder_1 + '/' + file_name[1] + '.npy'
data_1 = np.load(path_file_0)
data_1 = np.transpose(data_1,[3,4,2,1,0])
data_2 = np.load(path_file_1)
data_2 = np.transpose(data_2,[3,4,2,1,0])
size_x = np.size(data_1,0)
size_y = np.size(data_1,1)
size_z = np.size(data_1,2)
num_tp = np.size(data_1,3)
num_trial = np.size(data_1,4)
data = np.concatenate((data_1,data_2),axis = 4)
del data_1,data_2
data = data.reshape(size_x,size_y,size_z,num_tp,num_trial,num_channel,order = 'F')
print('load data done! size:')
print(np.shape(data))

# load atlas
file_folder_2 = file_folder_1[:-5]
atlas= tf.imread(atlas_path + '/' + file_folder_2 + '/' + atlas_name)
atlas = np.transpose(atlas,[1,2,0])
atlas = atlas[:,:,atlas_z_range]
print('load atlas done! size:')
print(np.shape(atlas))
for idx in br_index_1:
    atlas[atlas==idx] = br_index[0]

# regions
list_regions = np.unique(atlas)
list_regions = list_regions[list_regions>0]
num_regions = len(br_index)


############################
list_list_num_dim = np.zeros((num_channel_selected,num_odor_choice,num_regions,num_thresh,num_shuffle_choice))
list_list_accuracy = np.zeros((num_channel_selected,num_odor_choice,num_regions,num_thresh,cv_fold,num_shuffle_choice))
list_list_f1_weighted = np.zeros((num_channel_selected,num_odor_choice,num_regions,num_thresh,cv_fold,num_shuffle_choice))
list_list_AUC_weighted = np.zeros((num_channel_selected,num_odor_choice,num_regions,num_thresh,cv_fold,num_shuffle_choice))
list_list_accuracy_svm = np.zeros((num_channel_selected,num_odor_choice,num_regions,num_thresh,cv_fold,num_shuffle_choice))
list_list_stim = np.zeros((num_channel_selected,num_odor_choice,num_regions,num_thresh,len(stim),num_shuffle_choice))

result_each_fly_path = os.path.abspath(result_path + '/' + file_folder_list_2[fly_selected] + '/' + 'each_region_L-SRD-formal-real')
folder = os.path.exists(result_each_fly_path)
if not folder:
    os.makedirs(result_each_fly_path)
    
for j,channel_selected in enumerate(list_channel_selected):
    for k,odor_choice in enumerate(list_odor_choice):
        for m,region_selected in enumerate(br_index):
            if not region_selected in list_regions:
                continue
            print(br_name[m])
            for p in range(num_shuffle_choice):
                if if_shuffle[p]:
                    stim_2 = stim.copy()
                    random.shuffle(stim_2)
                else:
                    stim_2 = stim.copy()
                result_each_fly_path_1 = result_each_fly_path + '/' + if_shuffle_flag[p] + '/' + 'C_' + \
                                         str(channel_selected) + '_odor_choice_'+ str(odor_choice)
                folder = os.path.exists(result_each_fly_path_1)
                if not folder:
                    os.makedirs(result_each_fly_path_1)
                [x_origin,explained_variance_ratio] = pca_each_brain_region(data,stim_2,
                                                                            atlas,
                                          region_selected,br_name[m],channel_selected,
                                          odor_choice,dff0_thresh,pca_tp_range,
                                          if_plot_figure,if_save_figure,result_each_fly_path_1)
                # np.save(result_each_fly_path_1 + '/' + 'x_origin.npy',x_origin)
                for n,thresh_pca_exp_var_2 in enumerate(list_thresh_pca_exp_var_2):
                    [num_dim,list_accuracy,list_f1_weighted,list_AUC_weighted,list_accuracy_svm] = \
                        odor_classification_each_brain_region(x_origin,num_tp,np.sum(stim_2!=odor_choice),
                                          stim_2[stim_2!=odor_choice],explained_variance_ratio,br_name[m],
                                          odor_choice,thresh_pca_exp_var_2,lda_tp_selected,
                                          svm_tp_range,cv_fold,
                                          if_plot_figure,if_save_figure,
                                          result_each_fly_path_1)

                    list_list_num_dim[j,k,m,n,p] = num_dim
                    list_list_accuracy[j,k,m,n,:,p] = np.array(list_accuracy)
                    list_list_f1_weighted[j,k,m,n,:,p] = np.array(list_f1_weighted)
                    list_list_AUC_weighted[j,k,m,n,:,p] = np.array(list_AUC_weighted)
                    list_list_accuracy_svm[j,k,m,n,:,p] = np.array(list_accuracy_svm)
                    list_list_stim[j,k,m,n,:,p] = np.array(stim_2)

np.save(result_each_fly_path + '/' + 'list_regions_each_region' + '.npy',br_index)
np.save(result_each_fly_path + '/' + 'list_num_dim' + '.npy',list_list_num_dim)
np.save(result_each_fly_path + '/' + 'list_accuracy_each_region' + '.npy',list_list_accuracy)
np.save(result_each_fly_path + '/' + 'list_f1_weighted_each_region' + '.npy',list_list_f1_weighted)
np.save(result_each_fly_path + '/' + 'list_AUC_weighted_each_region' + '.npy',list_list_AUC_weighted)
np.save(result_each_fly_path + '/' + 'list_accuracy_svm_each_region' + '.npy',list_list_accuracy_svm)
np.save(result_each_fly_path + '/' + 'list_stim' + '.npy',list_list_stim)