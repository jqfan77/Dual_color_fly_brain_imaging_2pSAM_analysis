import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
def compute_mean_response_corr(data,atlas,stim_len,len_before_stim):
    num_x = np.size(data,0)
    num_y = np.size(data,1)
    num_z = np.size(data,2)
    num_trial = np.size(data,3)
    num_tp = np.size(data,4)
    atlas_mask = atlas>0
    mean_corr = np.zeros((num_x,num_y,num_z))
    b = np.concatenate((np.zeros(len_before_stim),\
                    np.ones(stim_len),\
                    np.zeros(num_tp-len_before_stim-stim_len)),\
                    axis = 0)
    bb = []
    for m in range(num_trial):
        bb.append(b)
    bb = np.array(bb)
    bb = bb.reshape(-1)
    plt.figure()
    plt.plot(bb)
    for i in range(num_x):
        for j in range(num_y):
            for k in range(num_z):
                if not atlas_mask[i,j,k]:
                    continue
                a = np.squeeze(data[i,j,k,:,:])
                aa = a.reshape(-1)
                result = pearsonr(aa,bb)
                if result.pvalue<0.05:
                    mean_corr[i,j,k] = result.statistic
        # print('row '+str(i)+ ' done!')
    return mean_corr