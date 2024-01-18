# input:
# dff0_stim: shape(size_x,size_y,size_z,num_tp,num_trial,num_channel)
# label: label of stims
# mask: shape(size_x,size_y,size_z)
# method_choice: 0-pca+lda+svm,1-pca+svm,2-svm
# output:
# acc, precision_weighted, recall_weighted, f1_weighted, auc_weighted,shape(size_x,size_y,size_z)
def compute_accuracy_map_svm(dff0_stim,label,odor_choice,mask,channel,win_x,win_y,win_z,cv_fold,kf_random_state,ifshuffle,method_choice = 1,pca_variance_threshold = 0.9):

    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn import svm
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import scale
    import warnings

    if odor_choice!=0:
        dff0_stim = dff0_stim[:,:,:,:,label!=odor_choice,:]
        label = label[label!=odor_choice]
    # parameters
    size_x = np.size(dff0_stim,0)
    size_y = np.size(dff0_stim,1)
    size_z = np.size(dff0_stim,2)
    num_tp = np.size(dff0_stim,3)
    num_trial = np.size(dff0_stim,4)
    num_channel = np.size(dff0_stim,5)
 
    size_x_win = int(np.ceil(size_x/win_x))
    size_y_win = int(np.ceil(size_y/win_y))
    size_z_win = int(np.ceil(size_z/win_z))

    acc = np.zeros((size_x_win,size_y_win,size_z_win))
    # precision_weighted = np.zeros((size_x_win,size_y_win,size_z_win))
    # recall_weighted = np.zeros((size_x_win,size_y_win,size_z_win))
    # f1_weighted = np.zeros((size_x_win,size_y_win,size_z_win))
    auc_weighted = np.zeros((size_x_win,size_y_win,size_z_win))

    # computing
    for i in range(size_x_win):
        for j in range(size_y_win):
            for k in range(size_z_win):
                x = dff0_stim[i*win_x:min((i+1)*win_x,size_x),j*win_y:min((j+1)*win_y,size_y),k*win_z:min((k+1)*win_z,size_z),:,:,:]
                x_mask = mask[i*win_x:min((i+1)*win_x,size_x),j*win_y:min((j+1)*win_y,size_y),k*win_z:min((k+1)*win_z,size_z)]
                x_mask = x_mask.reshape((np.size(x,0)*np.size(x,1)*np.size(x,2)),order = 'F')
                x = x.reshape((np.size(x,0)*np.size(x,1)*np.size(x,2),num_tp,num_trial,num_channel),order = 'F')
                x = x[x_mask == True,:,:,:]
                if len(x)==0:
                    continue
                if channel == 0:
                    x = np.transpose(x,[2,0,1,3])
                    x = x.reshape((num_trial,-1),order = 'F')
                elif channel == 1:
                    x = np.squeeze(x[:,:,:,0])
                    if x.ndim==2:
                        x = x[:,:,np.newaxis]
                        x = np.transpose(x,[2,0,1])
                    x = np.transpose(x,[2,0,1])
                    x = x.reshape((num_trial,-1),order = 'F')
                else:
                    x = np.squeeze(x[:,:,:,1])
                    if x.ndim==2:
                        x = x[:,:,np.newaxis]
                        x = np.transpose(x,[2,0,1])
                    x = np.transpose(x,[2,0,1])
                    x = x.reshape((num_trial,-1),order = 'F')

                # finally got x - > svm
                # pca
                if method_choice == 1:
                    pca = PCA()
                    x = scale(X=x,with_mean = True,with_std = True,copy = True)
                    x_pca = pca.fit_transform(x)
                    explained_variance_ratio = pca.explained_variance_ratio_
                    expr = 0
                    num_dim = 0
                    for m in range(len(explained_variance_ratio)):
                        expr = expr + explained_variance_ratio[m]
                        num_dim = m + 1
                        if expr>pca_variance_threshold: 
                            break
                    x_pca = x_pca[:,range(num_dim)]
                else:
                    x_pca = x
                # svm
                kf = KFold(n_splits=cv_fold, random_state=kf_random_state, shuffle=ifshuffle)
                clf = svm.SVC(probability=True)
                acc[i,j,k] = np.mean(cross_val_score(clf, x_pca, label, scoring='accuracy', cv=kf))
                # precision_weighted[i,j,k] = np.mean(cross_val_score(clf, x_pca, label, scoring='precision_weighted',cv=kf))
                # recall_weighted[i,j,k] = np.mean(cross_val_score(clf, x_pca, label, scoring='recall_weighted',cv=kf))
                # f1_weighted[i,j,k] = np.mean(cross_val_score(clf, x_pca, label, scoring='f1_weighted',cv=kf))
                auc_weighted[i,j,k] = np.mean(cross_val_score(clf, x_pca, label, scoring='roc_auc_ovo_weighted',cv=kf))
                warnings.filterwarnings('ignore')
    # return acc,precision_weighted,recall_weighted,f1_weighted,auc_weighted
    return acc,auc_weighted