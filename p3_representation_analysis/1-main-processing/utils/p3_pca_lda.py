import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_validate
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import f1_score
import random

def produce_brain_region(data,channel_selected,atlas,region_c1,region_c2,region_plus,stim_origin,odor_choice,
                         if_trial_shuffle,dff0_thresh,pca_tp_range):
    data = data[:,:,:,:,stim_origin!=odor_choice,:]
    stim_origin = stim_origin[stim_origin!=odor_choice]
    stim = stim_origin.copy()
    # trial shuffle
    if if_trial_shuffle:
        random.shuffle(stim)
    # channel
    if channel_selected != 0:
        data = data[:,:,:,:,:,channel_selected-1]
        if data.ndim==5:
            data = data[:,:,:,:,:,np.newaxis]
    num_channel = np.size(data,5)
    # dff0_thresh
    num_tp = np.size(data,3)
    num_trial = np.size(data,4)
    where_are_nan = np.isnan(data)
    where_are_inf = np.isinf(data)
    data[where_are_nan] = 0
    data[where_are_inf] = 0
    mean_data_reshape = np.std(data[:,:,:,pca_tp_range,:,:],3)
    mean_data_reshape = np.mean(mean_data_reshape,3)
    data = np.transpose(data,[0,1,2,5,3,4])
    data[mean_data_reshape>dff0_thresh,:,:] = 0
    data = np.transpose(data,[0,1,2,4,5,3])
    
    # each region: mean
    inds_region = np.linspace(1,86,86)
    num_region = len(inds_region)
    data_region = np.zeros((num_region,num_tp,num_trial,num_channel))
    for i,ind_region in enumerate(inds_region):
        a = (atlas==ind_region)
        if np.sum(a)>0:
            b = data[a,:,:,:]
            data_region[i,:,:,:] = np.mean(b,0)
    data_region = np.transpose(data_region,[1,2,0,3])
    if channel_selected==1:
        data_region = data_region[:,:,region_c1,:]
        data_region = data_region.reshape((num_tp*num_trial,-1),order = 'F')
    elif channel_selected==2:
        data_region = data_region[:,:,region_c2,:]
        data_region = data_region.reshape((num_tp*num_trial,-1),order = 'F')
    else:
        data_region_1 = np.squeeze(data_region[:,:,region_c1,0])
        data_region_2 = np.squeeze(data_region[:,:,region_plus,1])
        data_region = np.concatenate((data_region_1,data_region_2),axis = 2)
        data_region = data_region.reshape((num_tp*num_trial,-1),order = 'F')
    print('data_region done!')
    return data_region,stim,num_tp,num_trial
    
def odor_classification_brain_region(data_region,stim,num_tp,num_trial,thresh_pca_exp_var_2,
                                     odor_choice,cv_fold,pca_tp_range,lda_tp_selected,svm_tp_range,
                                     if_plot_figure = False,if_save_figure = False,savepath = ''):
    c = ['b','g','m']
    list_accuracy = []
    list_f1_weighted = []
    list_AUC_weighted = []
    list_accuracy_svm = []
    x_origin = data_region

    # pca
    pca = PCA()
    x_origin_1 = x_origin.reshape((num_tp,num_trial,-1),order = 'F')
    x_origin_1 = x_origin_1[pca_tp_range,:,:]
    x_origin_1 = x_origin_1.reshape((len(pca_tp_range)*num_trial,-1),order = 'F')
    x_origin_1 = scale(X=x_origin_1,with_mean=True,with_std=True,copy=True)
    x_origin = scale(X=x_origin,with_mean=True,with_std=True,copy=True)
    pca.fit(x_origin_1)
    x_origin = pca.transform(x_origin)
    # plot pca projection
    xx = x_origin.reshape((num_tp,num_trial,-1),order = 'F')
    xx = xx[:,:,range(2)]
    xx = np.transpose(xx,[1,0,2])
    plt.figure()
    for i in range(num_trial):
        plt.plot(np.squeeze(xx[i,:,0]),np.squeeze(xx[i,:,1]),color = c[stim[i]-1])
    if if_save_figure:
        plt.savefig(savepath + '/' + 'pca_projection' + '.pdf')
        plt.savefig(savepath + '/' + 'pca_projection' + '.png')
        np.save(savepath + '/' + 'pca_projection' + '.npy',x_origin)
        np.save(savepath + '/' + 'stim' + '.npy',stim)
    if if_plot_figure:
        plt.show()
    plt.close()
    # first n dims    
    explained_variance_ratio = pca.explained_variance_ratio_
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
    x_svm = x.reshape((num_tp,num_trial,-1),order = 'F')
    x_svm = x_svm[svm_tp_range,:,:]
    print(np.shape(x_svm))
    x_svm = np.transpose(x_svm,[1,0,2])
    x_svm = x_svm.reshape((num_trial,-1),order = 'F')
    x_svm = scale(X=x_svm,with_mean=True,with_std=True,copy=True)
    clf = svm.SVC()
    cv_results = cross_validate(clf, x_svm, stim, cv=kf)
    list_accuracy_svm.append(cv_results['test_score'])

    #### LDA and SVM
    list_accuracy_1 = []
    list_f1_weighted_1 = []
    list_AUC_weighted_1 = []
    kk = 1
    x_lda = x.reshape((num_tp,num_trial,-1),order = 'F')
    for train, test in kf.split(np.linspace(0,num_trial-1,num_trial)):
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

        #### LDA
        clf = LDA()
        flag_x_lda = np.squeeze(x_lda[lda_tp_selected,train_origin,:])
        flag_x_lda = flag_x_lda.reshape((len(train_origin),-1),order = 'F')
        clf.fit(flag_x_lda, stim[train_origin])
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
                plt.savefig(savepath + '/' + 'training_set'+ '_expr_thresh_' + 
                            str(expr_thresh) + '_fold_'+ str(kk) + '.pdf')
                plt.savefig(savepath + '/' + 'training_set'+ '_expr_thresh_' + 
                            str(expr_thresh) + '_fold_'+ str(kk) + '.png')
                np.save(savepath + '/' + 'training_set'+ '_expr_thresh_' + 
                            str(expr_thresh) + '_fold_'+ str(kk) + '.npy',x_train_1)
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
                plt.savefig(savepath + '/' + 'testing_set' + '_expr_thresh_' + 
                            str(expr_thresh) + '_fold_'+ str(kk) + '.pdf')
                plt.savefig(savepath + '/' + 'testing_set' + '_expr_thresh_' + 
                            str(expr_thresh) + '_fold_'+ str(kk) + '.png')
                np.save(savepath + '/' + 'testing_set' + '_expr_thresh_' + 
                            str(expr_thresh) + '_fold_'+ str(kk) + '.npy',x_test_1)
            if if_plot_figure:
                plt.show()
            plt.close()
        # svm
        x_train_1 = x_train_1[svm_tp_range,:,:]
        x_test_1 = x_test_1[svm_tp_range,:,:]
        x_train_2 = np.transpose(x_train_1,[1,0,2])
        x_train_2 = x_train_2.reshape((np.size(x_train_2,0),np.size(x_train_2,1)*np.size(x_train_2,2)),order = 'F')
        x_test_2 = np.transpose(x_test_1,[1,0,2])
        x_test_2 = x_test_2.reshape((np.size(x_test_2,0),np.size(x_test_2,1)*np.size(x_test_2,2)),order = 'F')
        x_scale = np.concatenate((x_train_2,x_test_2),axis = 0)
        x_scale = scale(X=x_scale,with_mean=True,with_std=True,copy=True)
        x_train_2 = x_scale[0:len(train_origin),:]
        x_test_2 = x_scale[len(train_origin):num_trial,:]
        clf = svm.SVC().fit(x_train_2, stim[train_origin])
        # indices
        acc = clf.score(x_test_2, stim[test_origin])
        y = clf.predict(x_test_2)
        f1_weighted = f1_score(stim[test_origin],y,average = 'weighted')
        area = 0
        if odor_choice==0:
            dff = clf.decision_function(x_test_2)
            dff_sum = (np.sum(clf.decision_function(x_test_2),1))
            dff_sum = dff_sum[:, np.newaxis]
            area = AUC(stim[test_origin],dff/dff_sum,average='weighted',multi_class = 'ovo') 
        else:
            dff = clf.decision_function(x_test_2)
            area = AUC(stim[test_origin],dff) 
        list_accuracy_1.append(acc)
        list_f1_weighted_1.append(f1_weighted)
        list_AUC_weighted_1.append(area)
        kk = kk+1
    

    list_accuracy.append(list_accuracy_1)
    list_f1_weighted.append(list_f1_weighted_1)
    list_AUC_weighted.append(list_AUC_weighted_1) 
    return data_region,list_accuracy,list_f1_weighted,list_AUC_weighted,list_accuracy_svm
    