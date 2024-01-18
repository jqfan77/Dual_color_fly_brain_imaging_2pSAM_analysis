import numpy as np
import matplotlib.pyplot as plt
import os
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_validate
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import f1_score
import random



# pca in each block
def pca_block(a,channel_selected,num_tp,num_trial,mask,thresh_pca_exp_var_1,pca_tp_range,pca_data_c0,block_dim):
    a_c0 = a[:,:,:,:,:,channel_selected]
    a_c0 = a_c0.reshape((-1,num_tp*num_trial),order = 'F')
    a_c0 = np.transpose(a_c0,[1,0])
    a_c0 = a_c0[:,mask]
    b_c0 = a[:,:,:,:,:,channel_selected]
    b_c0 = b_c0[:,:,:,pca_tp_range,:]
    num_tp_1 = len(pca_tp_range)
    b_c0 = b_c0.reshape((-1,num_tp_1*num_trial),order = 'F')
    b_c0 = np.transpose(b_c0,[1,0])
    b_c0 = b_c0[:,mask]
    if np.size(a_c0,1)>0:
        a_c0=scale(X=a_c0,with_mean=True,with_std=True,copy=True)
        b_c0=scale(X=b_c0,with_mean=True,with_std=True,copy=True)
        pca = PCA()
        pca.fit(b_c0)
        y = pca.transform(a_c0)
        explained_variance_ratio = pca.explained_variance_ratio_
        expr = 0
        num_dim = 0
        for m in range(len(explained_variance_ratio)):
            expr = expr + explained_variance_ratio[m]
            num_dim = m+1
            if expr>thresh_pca_exp_var_1:
                break
        if not len(pca_data_c0):
            pca_data_c0 = y[:,range(num_dim)]
        else:
            pca_data_c0  = np.concatenate((pca_data_c0,y[:,range(num_dim)]),axis = 1)  
        block_dim.append(num_dim)
    return pca_data_c0,block_dim
# block-wise pca for voxel-level multiple-brain-region classification; and pca using the consolidated PCs of each block
def fly_pca_block(data,atlas,atlas_c1,atlas_c2,atlas_plus,stim,channel_selected,odor_choice,
                  thresh_pca_exp_var_1,block_size_ratio_xy,block_size_ratio_z,dff0_thresh,
                  pca_tp_range,svm_tp_range,if_plot_figure = False,if_save_figure = False,savepath = '',savepath_common = ''):
# data: size_x,size_y,size_z,num_tp,num_trial,num_channel
# stim: num_trial
# atlas: size_x,size_y,size_z

    # stim selected
    data = data[:,:,:,:,stim!=odor_choice,:]
    stim = stim[stim!=odor_choice]
    # trial shuffle
    stim_shuffled = stim.copy()
    random.shuffle(stim_shuffled)
    stim_origin = stim.copy()
    # channel
    if channel_selected != 0:
        data = data[:,:,:,:,:,channel_selected-1]
    # imresize-downsample
    if data.ndim == 5:
        data = data[:,:,:,:,:,np.newaxis]
    num_channel = np.size(data,5)
    num_tp = np.size(data,3)
    num_trial = np.size(data,4)
    size_x = np.size(data,0)
    size_y = np.size(data,1)
    size_z = np.size(data,2)
    plt.imshow(atlas.max(2))
    plt.show()
    atlas_mask = atlas>0
    atlas_mask_c1 = atlas_c1>0
    atlas_mask_c2 = atlas_c2>0
    atlas_mask_plus = atlas_plus>0

    
    # show mean response
    where_are_nan = np.isnan(data)
    data[where_are_nan] = 0
    del where_are_nan
    where_are_inf = np.isinf(data)
    data[where_are_inf] = 0
    del where_are_inf
    mean_data_reshape = np.std(np.mean(data[:,:,:,svm_tp_range,:,:],4),3)
    for i in range(num_channel): 
        m = np.squeeze(mean_data_reshape[:,:,:,i])
        data[m>dff0_thresh,:,:,i] = 0
    mean_data_reshape[mean_data_reshape>dff0_thresh] = 0
    for i in range(num_channel): 
        a = np.squeeze(mean_data_reshape[:,:,:,i])
        a = a*atlas_mask
        plt.imshow(a.max(2))
        plt.colorbar()
        if if_save_figure:
            plt.savefig(savepath + '/' + 'max_projection_mean_response_C' + str(i) + '.pdf')
            plt.savefig(savepath + '/' + 'max_projection_mean_response_C' + str(i) + '.png')
            np.save(savepath + '/' + 'mean_response_C' + str(i) + '.npy',a)
        plt.show()
        plt.close()

    # show filtered voxels
    for i in range(num_channel):
        a = np.squeeze(mean_data_reshape[:,:,:,i])
        if channel_selected==0:
            if i==0:
                the_mask = atlas_mask_c1
            else:
                the_mask = atlas_mask_plus
        elif channel_selected==1:
            the_mask = atlas_mask_c1
        else:
            the_mask = atlas_mask_c2
        a = a*atlas_mask
        loc_voxels = np.argwhere(the_mask==True)
        plt.imshow(a.max(2))
        plt.colorbar()
        plt.scatter(loc_voxels[:,1],loc_voxels[:,0],s = 3,c='r')
        if if_save_figure:
            plt.savefig(savepath + '/' + 'max_projection_mean_response_C' + str(i) + '_filtered_voxels' +'.pdf')
            plt.savefig(savepath + '/' + 'max_projection_mean_response_C' + str(i) + '_filtered_voxels' +'.png')
            np.save(savepath + '/' + 'loc_voxels_C' + str(i) + '.npy',loc_voxels)
        plt.show()
        plt.close()

    del mean_data_reshape,loc_voxels,a
    # pca for each block
    pca_size_xy = math.floor(size_x*block_size_ratio_xy)
    pca_size_z = math.floor(size_z*block_size_ratio_z)
    num_block_xy = math.ceil(size_x/pca_size_xy)
    num_block_z = math.ceil(size_z/pca_size_z)
    if channel_selected != 0:
        if channel_selected==1 and os.path.exists(savepath_common + '/' + 'pca_data_c0.npy'):
            pca_data_c0 = np.load(savepath_common + '/' + 'pca_data_c0.npy')
            block_dim = np.load(savepath_common + '/' + 'block_dim_c0.npy')
        else:
            if data.ndim == 5:
                data = data[:,:,:,:,:,np.newaxis]
            if channel_selected==1:
                atlas_mask = atlas_mask_c1
            else:
                atlas_mask = atlas_mask_c2
            pca_data_c0 = []
            block_dim = []
            for i in range(num_block_xy):
                range_x = range(i*pca_size_xy,min((i+1)*pca_size_xy,size_x))
                mask_1 = atlas_mask[range_x,:,:]
                if np.sum(mask_1)==0:
                    print('Row ' + str(i) + 'Done!')
                    continue
                for j in range(num_block_xy):
                    range_y = range(j*pca_size_xy,min((j+1)*pca_size_xy,size_y))
                    mask_2 = mask_1[:,range_y,:]
                    if np.sum(mask_2)==0:
                        continue
                    for k in range(num_block_z):
                        range_z = range(k*pca_size_z,min((k+1)*pca_size_z,size_z))
                        mask = mask_2[:,:,range_z]
                        if np.sum(mask)==0:
                            continue
                        a = data[range_x,:,:,:,:,:]
                        a = a[:,range_y,:,:,:,:]
                        a = a[:,:,range_z,:,:,:]
                        mask = mask.reshape(-1,order = 'F')
                        [pca_data_c0,block_dim] = pca_block(a,0,num_tp,num_trial,mask,thresh_pca_exp_var_1,pca_tp_range,pca_data_c0,block_dim)
                print('Row ' + str(i) + 'Done!')
        print('C'+ str(channel_selected-1) +' Dim:') 
        print(np.size(pca_data_c0,1)) 
        x_origin = pca_data_c0
    else:
        pca_data_c0 = []
        pca_data_c1 = []
        block_dim_c0 = []
        block_dim_c1 = []
        for i in range(num_block_xy):
            range_x = range(i*pca_size_xy,min((i+1)*pca_size_xy,size_x))
            mask_1 = atlas_mask_c1[range_x,:,:]
            if np.sum(mask_1)==0:
                print('Row ' + str(i) + 'Done!')
                continue
            for j in range(num_block_xy):
                range_y = range(j*pca_size_xy,min((j+1)*pca_size_xy,size_y))
                mask_2 = mask_1[:,range_y,:]
                if np.sum(mask_2)==0:
                    continue
                for k in range(num_block_z):
                    range_z = range(k*pca_size_z,min((k+1)*pca_size_z,size_z))
                    mask = mask_2[:,:,range_z]
                    if np.sum(mask)==0:
                        continue
                    a = data[range_x,:,:,:,:,:]
                    a = a[:,range_y,:,:,:,:]
                    a = a[:,:,range_z,:,:,:]
                    # channel 1
                    mask = mask.reshape(-1,order = 'F')
                    [pca_data_c0,block_dim_c0] = pca_block(a,0,num_tp,num_trial,mask,thresh_pca_exp_var_1,pca_tp_range,pca_data_c0,block_dim_c0)
                    # channel 2
                    mask = atlas_mask_plus[range_x,:,:]
                    mask = mask[:,range_y,:]
                    mask = mask[:,:,range_z]
                    mask = mask.reshape(-1,order = 'F')
                    [pca_data_c1,block_dim_c1] = pca_block(a,1,num_tp,num_trial,mask,thresh_pca_exp_var_1,pca_tp_range,pca_data_c1,block_dim_c1)
            print('Row ' + str(i) + 'Done!')
        print('C0 Dim:') 
        print(np.size(pca_data_c0,1)) 
        print('C1 Dim:')
        print(np.size(pca_data_c1,1))
        x_origin=np.concatenate((pca_data_c0,pca_data_c1),axis = 1)
        block_dim_c0 = np.array(block_dim_c0)
        block_dim_c0 = block_dim_c0.reshape((-1,1),order = 'F')
        block_dim_c1 = np.array(block_dim_c1)
        block_dim_c1 = block_dim_c1.reshape((-1,1),order = 'F')
        block_dim = np.concatenate((block_dim_c0,block_dim_c1),axis = 0)
        np.save(savepath_common + '/' + 'pca_data_c0.npy',pca_data_c0)
        np.save(savepath_common + '/' + 'block_dim_c0.npy',block_dim_c0)

    pca = PCA()
    x_origin_1 = x_origin.reshape((num_tp,num_trial,-1),order = 'F')
    x_origin_1 = x_origin_1[pca_tp_range,:,:]
    x_origin_1 = x_origin_1.reshape((len(pca_tp_range)*num_trial,-1),order = 'F')
    pca.fit(x_origin_1)
    x_pca = pca.transform(x_origin)
    # plot pca projection
    xx = x_pca.reshape((num_tp,num_trial,-1),order = 'F')
    xx = xx[:,:,range(2)]
    xx = np.transpose(xx,[1,0,2])
    plt.figure()
    c = ['b','g','m']
    for i in range(num_trial):
        plt.plot(np.squeeze(xx[i,:,0]),np.squeeze(xx[i,:,1]),color = c[stim[i]-1])
    if if_save_figure:
        plt.savefig(savepath + '/' + 'pca_projection' + '.pdf')
        plt.savefig(savepath + '/' + 'pca_projection' + '.png')
        np.save(savepath + '/' + 'pca_projection' + '.npy',x_pca)
        # np.save(savepath + '/' + 'stim' + '.npy',stim)
    if if_plot_figure:
        plt.show()
    plt.close()
            
    explained_variance_ratio = pca.explained_variance_ratio_
    
    return x_origin,block_dim,stim_origin,stim_shuffled,num_tp,num_trial,x_pca,explained_variance_ratio
# lda and svm of the trials
def lda_trial(x,x_lda,train,test,stim,num_tp,num_trial,expr_thresh,kk,odor_choice,c,
              lda_tp_selected,svm_tp_range,if_plot_figure,if_save_figure,savepath):

    train_origin = (np.array(train)).astype(int)
    test_origin = (np.array(test)).astype(int)

    flag_train = []
    for j in range(len(train)):
        L = np.linspace(int(train[j]*num_tp),int((train[j]+1)*num_tp-1),num_tp).tolist()
        flag_train.extend(L)
    flag_test = []
    for j in range(len(test)):
        L = np.linspace(int(test[j]*num_tp),int((test[j]+1)*num_tp-1),num_tp).tolist()
        flag_test.extend(L)
    

    train = np.array(flag_train)
    test = np.array(flag_test)
    train = train.astype(int)
    test = test.astype(int) 

    ### LDA
    clf = LDA()
    flag_x_lda = np.squeeze(x_lda[lda_tp_selected,train_origin,:])
    flag_x_lda = flag_x_lda.reshape((len(train_origin),-1),order = 'F')
    clf.fit(flag_x_lda,stim[train_origin])
    x_train = clf.transform(x[train,:])
    x_train_1 = x_train.reshape((num_tp,int(len(train)/num_tp),-1),order = 'F')
    y_1 = stim[train_origin]
    # train
    the_label = ['OCT','MCH','EA']
    if np.size(x_train_1,2)>1:
        plt.figure()
        for j in range(np.size(x_train_1,1)):
            a = np.squeeze(x_train_1[:,j,:])
            plt.plot(a[:,0],a[:,1],color = c[y_1[j]-1],label = the_label[y_1[j]-1])
        if if_save_figure:
            plt.savefig(savepath + '/' + 'training_set' + '_expr_thresh_' + str(expr_thresh) + '_fold_'+ str(kk) + '.pdf')
            plt.savefig(savepath + '/' +  'training_set'+ '_expr_thresh_' + str(expr_thresh) + '_fold_'+ str(kk) + '.png')
            np.save(savepath + '/' +  'training_set'+ '_expr_thresh_' + str(expr_thresh) + '_fold_'+ str(kk) + '.npy',x_train_1)
        if if_plot_figure:
            plt.show()
        plt.close()
    # test
    x_test = clf.transform(x[test,:])
    x_test_1 = x_test.reshape((num_tp,int(len(test)/num_tp),-1),order = 'F')
    y_1 = stim[test_origin]
    if np.size(x_test_1,2)>1:
        plt.figure()
        for j in range(np.size(x_test_1,1)):
            a = np.squeeze(x_test_1[:,j,:])
            plt.plot(a[:,0],a[:,1],color = c[y_1[j]-1],label = the_label[y_1[j]-1])
        if if_save_figure:
            plt.savefig(savepath + '/' + 'testing_set' + '_expr_thresh_' + str(expr_thresh) + '_fold_'+ str(kk) + '.pdf')
            plt.savefig(savepath + '/' + 'testing_set' + '_expr_thresh_' + str(expr_thresh) + '_fold_'+ str(kk) + '.png')
            np.save(savepath + '/' + 'testing_set' + '_expr_thresh_' + str(expr_thresh) + '_fold_'+ str(kk) + '.npy',x_test_1)
        if if_plot_figure:
            plt.show()
        plt.close()
    
    x_train_1 = x_train_1[svm_tp_range,:,:]
    x_test_1 = x_test_1[svm_tp_range,:,:]
    x_train_2 = np.transpose(x_train_1,[1,0,2])
    x_train_2 = x_train_2.reshape((np.size(x_train_2,0),np.size(x_train_2,1)*np.size(x_train_2,2)),order = 'F')
    x_test_2 = np.transpose(x_test_1,[1,0,2])
    x_test_2 = x_test_2.reshape((np.size(x_test_2,0),np.size(x_test_2,1)*np.size(x_test_2,2)),order = 'F')
    
    # svm
    x_scale = np.concatenate((x_train_2,x_test_2),axis = 0)
    x_scale = scale(X=x_scale,with_mean=True,with_std=True,copy=True)
    x_train_2 = x_scale[0:len(train_origin),:]
    x_test_2 = x_scale[len(train_origin):num_trial,:]
    clf = svm.SVC().fit(x_train_2, stim[train_origin])
    acc = clf.score(x_test_2, stim[test_origin])
    y = clf.predict(x_test_2)
    f1_weighted = f1_score(stim[test_origin],y,average = 'weighted')
    # AUC
    area = 0
    if odor_choice==0:
        dff = clf.decision_function(x_test_2)
        dff_sum = (np.sum(clf.decision_function(x_test_2),1))
        dff_sum = dff_sum[:, np.newaxis]
        area = AUC(stim[test_origin],dff/dff_sum,average='weighted',multi_class = 'ovo') 
    else:
        dff = clf.decision_function(x_test_2)
        area = AUC(stim[test_origin],dff) 
    return acc, f1_weighted, area
# voxel-level multiple-brain-region classification
def odor_classification(x_origin,explained_variance_ratio,stim,
                        odor_choice,num_tp,num_trial,thresh_pca_exp_var_2,
                        lda_tp_selected,svm_tp_range,cv_fold,if_plot_figure = False,if_save_figure = False,
                        savepath = ''):

    # LDA and SVM
    list_accuracy = []
    list_f1_weighted = []
    list_AUC_weighted = []
    list_accuracy_svm = []
    c = ['b','g','m']
    # take first n PCs
    expr_thresh = thresh_pca_exp_var_2
    expr = 0
    num_dim = 0
    for m in range(len(explained_variance_ratio)):
        expr = expr + explained_variance_ratio[m]
        num_dim = m+1
        if expr>expr_thresh:
            break
    x = x_origin[:,range(num_dim)]
    print(np.shape(x))

    
    kf = KFold(n_splits=cv_fold,shuffle = True,random_state = 5)
    #### only svm
    x_svm = x.reshape((num_tp,num_trial,num_dim),order = 'F')
    x_svm = x_svm[svm_tp_range,:,:]
    print(np.shape(x_svm))
    x_svm = np.transpose(x_svm,[1,0,2])
    x_svm = x_svm.reshape((num_trial,len(svm_tp_range)*num_dim),order = 'F')
    x_svm = scale(X=x_svm,with_mean=True,with_std=True,copy=True)
    clf = svm.SVC()
    cv_results = cross_validate(clf, x_svm, stim, cv=kf)
    list_accuracy_svm.append(cv_results['test_score'])


    #### LDA
    list_accuracy_1 = []
    list_f1_weighted_1 = []
    list_AUC_weighted_1 = []
    kk = 1
    x_lda = x.reshape((num_tp,num_trial,num_dim),order = 'F')
    for train, test in kf.split(np.linspace(0,num_trial-1,num_trial)):
        [acc, f1_weighted, area] = lda_trial(x,x_lda,train,test,stim,num_tp,num_trial,expr_thresh,kk,odor_choice,c,
                                             lda_tp_selected,svm_tp_range,if_plot_figure,if_save_figure,savepath)
        # update list
        list_accuracy_1.append(acc)
        list_f1_weighted_1.append(f1_weighted)
        list_AUC_weighted_1.append(area)
        kk = kk+1
    

    list_accuracy.append(list_accuracy_1)
    list_f1_weighted.append(list_f1_weighted_1)
    list_AUC_weighted.append(list_AUC_weighted_1) 
    return num_dim,list_accuracy,list_f1_weighted,list_AUC_weighted,list_accuracy_svm
