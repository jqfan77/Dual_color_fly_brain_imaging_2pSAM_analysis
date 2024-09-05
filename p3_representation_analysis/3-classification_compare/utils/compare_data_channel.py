import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import matplotlib as mpl
import math
import os 

def compare_data_channel(result_path,file_folder_name,file_name,fly_selected,flag,channel,the_label,ymin,ymax,dim_thresh,
                        if_shuffle,the_color,if_save,figure_save_path,figure_save_name,test_way):
    file_folders = os.listdir(result_path)
    num_fly = len(fly_selected)
    num_channel_choice = len(channel)
    list_acc = []
    for i,id in enumerate(fly_selected):
        file_folder = file_folders[id]
        # load data
        the_path = result_path + '/' + file_folder + '/' + file_folder_name
        acc = np.load(the_path + '/' + file_name)
        the_acc = np.mean(np.squeeze(acc),2)
        # load num_dim 
        num_dim = np.load(the_path+ '/'+'list_list_num_dim.npy')
        num_dim = np.squeeze(num_dim)
        if if_shuffle:
            the_acc = np.squeeze(the_acc[:,:,1])
            num_dim = np.squeeze(num_dim[:,:,1])
        else:
            the_acc = np.squeeze(the_acc[:,:,0])
            num_dim = np.squeeze(num_dim[:,:,0])
        
        # compute the_the_acc
        the_the_acc = np.zeros(num_channel_choice)
        for j in range(num_channel_choice):
            a = np.squeeze(the_acc[channel[j],:])
            n = np.squeeze(num_dim[channel[j],:])
            the_flag = 0
            for k in range(len(n)):
                if n[k]>=dim_thresh:
                    the_flag = k
                    break
            the_the_acc[j] = a[the_flag]

        list_acc.append(the_the_acc)
    list_acc = np.array(list_acc)
    list_acc = list_acc*100
    list_acc = list_acc[:,::-1]
    # plot
    the_mean = np.mean(list_acc,0)
    the_std = np.std(list_acc,0)/math.sqrt(np.size(list_acc,0))
    plt.figure(figsize = (2,2.5))
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for k in range(num_fly):
        plt.plot(channel,list_acc[k,:],c = the_color,linewidth=1,alpha = 0.2)
    plt.errorbar(channel,the_mean,yerr=the_std,ecolor=the_color,elinewidth=1,marker='.',mfc=the_color,
	mec=the_color,mew=1,ms=1,alpha=1,capsize=5,capthick=3,color=the_color, linewidth=2)
    plt.ylim((ymin,ymax))
    plt.xticks(channel,the_label[::-1])
    plt.ylabel(flag + ' (%)')
    for j in range(num_channel_choice-1):
        if test_way == 'wilcoxon':
            res = wilcoxon(list_acc[:,num_channel_choice-1],list_acc[:,j],alternative = 'greater')
        else:
            res = mannwhitneyu(list_acc[:,num_channel_choice-1],list_acc[:,j])
        p = res.pvalue
        print(test_way + ' p:')
        print(p)
        if p<0.05 and p>=0.01:
            plt.text(j,ymax,'*',verticalalignment = 'center', horizontalalignment = 'center')
        elif p<0.01 and p>0.001:
            plt.text(j,ymax,'**',verticalalignment = 'center', horizontalalignment = 'center')
        elif p<0.001 and p>=0.0001:
            plt.text(j,ymax,'***',verticalalignment = 'center', horizontalalignment = 'center') 
        elif p<0.0001:
            plt.text(j,ymax,'****',verticalalignment = 'center', horizontalalignment = 'center') 
    if if_save:
        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42
        plt.savefig(figure_save_path + '/' + figure_save_name +'.pdf',dpi = 300,bbox_inches = 'tight')
        plt.savefig(figure_save_path + '/' + figure_save_name +'.png',dpi = 300,bbox_inches = 'tight')
    plt.show()
    