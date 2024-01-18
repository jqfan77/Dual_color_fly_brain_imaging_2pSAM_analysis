import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_validate
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import f1_score

def pca_each_brain_region(x_origin,stim,atlas_eroded,region_selected,region_name,channel_selected,
                                          odor_choice,dff0_thresh,pca_tp_range,
                                          if_plot_figure = False,if_save_figure = False,savepath = ''):
    c = ['b','g','m']
    savepath_1 = savepath + '/' + region_name
    folder = os.path.exists(savepath_1)
    if not folder:
        os.makedirs(savepath_1)
    # select brain region
    x_origin_mean = np.std(np.mean(x_origin[:,:,:,pca_tp_range,:,:],4),3)
    x_origin_mean = np.max(x_origin_mean,3)
    a = atlas_eroded.copy()
    a[x_origin_mean>dff0_thresh] = 0
    x_origin = x_origin[a==region_selected,:,:,:]
    
    # odor choice
    x_origin = x_origin[:,:,stim!=odor_choice,:]
    stim = stim[stim!=odor_choice]
    num_tp = np.size(x_origin,1)
    num_trial = np.size(x_origin,2)
    label = stim
    for i in range(num_tp-1):
        label = np.concatenate((label,stim),axis = 0)
    label = label.reshape(num_trial,num_tp,order = 'F')
    label = np.transpose(label,[1,0])
    label = label.reshape(num_trial*num_tp,order = 'F')
    # channel
    if channel_selected!=0:
        x_origin = x_origin[:,:,:,channel_selected-1]
        if x_origin.ndim==3:
            x_origin = x_origin[:,:,:,np.newaxis]
    x_origin = np.transpose(x_origin,[1,2,0,3])
    x_origin = x_origin.reshape((num_tp*num_trial,-1),order = 'F')

    # PCA
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
        plt.savefig(savepath_1 + '/' + 'pca_projection' + '.pdf')
        plt.savefig(savepath_1 + '/' + 'pca_projection' + '.png')
        np.save(savepath_1 + '/' + 'pca_projection' + '.npy',x_origin)
        np.save(savepath_1 + '/' + 'stim' + '.npy',stim)
    if if_plot_figure:
        plt.show()
    plt.close()
    
    explained_variance_ratio = pca.explained_variance_ratio_
    # print(np.sum(explained_variance_ratio))

    return x_origin,explained_variance_ratio

def odor_classification_each_brain_region(x_origin,num_tp,num_trial,stim,explained_variance_ratio,region_name,
                                          odor_choice,thresh_pca_exp_var_2,lda_tp_selected,
                                          svm_tp_range,cv_fold,
                                          if_plot_figure = False,if_save_figure = False,
                                          savepath = ''):
    c = ['b','g','m']
    the_label = ['OCT','MCH','EA']
    savepath_1 = savepath + '/' + region_name
    folder = os.path.exists(savepath_1)
    if not folder:
        os.makedirs(savepath_1)
    # first n dims
    expr_thresh = thresh_pca_exp_var_2
    expr = 0
    num_dim = 0
    for m in range(len(explained_variance_ratio)):
        expr = expr + explained_variance_ratio[m]
        num_dim = m+1
        if expr>expr_thresh:
            break
    x = x_origin[:,range(num_dim)]
    # print(np.shape(x))

    list_accuracy = []
    list_f1_weighted = []
    list_AUC_weighted = []
    list_accuracy_svm = []
    kf = KFold(n_splits=cv_fold,shuffle = True,random_state = 5)
    #### only svm
    x_svm = x.reshape((num_tp,num_trial,num_dim),order = 'F')
    x_svm = x_svm[svm_tp_range,:,:]
    # print(np.shape(x_svm))
    x_svm = np.transpose(x_svm,[1,0,2])
    x_svm = x_svm.reshape((num_trial,len(svm_tp_range)*num_dim),order = 'F')
    x_svm = scale(X=x_svm,with_mean=True,with_std=True,copy=True)
    clf = svm.SVC()
    cv_results = cross_validate(clf, x_svm, stim, cv=kf)
    list_accuracy_svm.append(cv_results['test_score'])
    #### LDA and SVM
    kk = 1
    x_lda = x.reshape((num_tp,num_trial,num_dim),order = 'F')
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
        if np.size(x_train_1,2)>1:
            plt.figure()
            for j in range(np.size(x_train_1,1)):
                a = np.squeeze(x_train_1[:,j,:])
                plt.plot(a[:,0],a[:,1],color = c[y_1[j]-1],label = the_label[y_1[j]-1])
            if if_save_figure:
                plt.savefig(savepath_1 + '/' + 'training_set' + '_expr_thresh_' + str(expr_thresh) + '_fold_'+ str(kk) + '.pdf')
                plt.savefig(savepath_1 + '/' +  'training_set' +'_expr_thresh_' + str(expr_thresh) + '_fold_'+ str(kk) + '.png')
                np.save(savepath_1 + '/' +  'training_set' +'_expr_thresh_' + str(expr_thresh) + '_fold_'+ str(kk) + '.npy',x_train_1)
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
                plt.savefig(savepath_1 + '/' + 'testing_set' + '_expr_thresh_' + str(expr_thresh) + '_fold_'+ str(kk) + '.pdf')
                plt.savefig(savepath_1 + '/' +  'testing_set' +'_expr_thresh_' + str(expr_thresh) + '_fold_'+ str(kk) + '.png')
                np.save(savepath_1 + '/' +  'testing_set' +'_expr_thresh_' + str(expr_thresh) + '_fold_'+ str(kk) + '.npy',x_test_1)
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
        acc = clf.score(x_test_2, stim[test_origin])
        y = clf.predict(x_test_2)
        f1_weighted = f1_score(stim[test_origin],y,average = 'weighted')
        # indices
        area = 0
        if odor_choice==0:
            dff = clf.decision_function(x_test_2)
            dff_sum = (np.sum(clf.decision_function(x_test_2),1))
            dff_sum = dff_sum[:, np.newaxis]
            area = AUC(stim[test_origin],dff/dff_sum,average='weighted',multi_class = 'ovo') 
        else:
            dff = clf.decision_function(x_test_2)
            area = AUC(stim[test_origin],dff) 
        list_accuracy.append(acc)
        list_f1_weighted.append(f1_weighted)
        list_AUC_weighted.append(area)
        kk = kk+1
    

    return num_dim,list_accuracy,list_f1_weighted,list_AUC_weighted,list_accuracy_svm
