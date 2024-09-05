import numpy as np
from sklearn.model_selection import KFold
import matplotlib as mpl
import matplotlib.pyplot as plt
import copy
from matplotlib.collections import LineCollection
from scipy.interpolate import interp1d
import os
import math
from scipy.stats import wilcoxon
from scipy.stats import mannwhitneyu
import pandas as pd
from scipy import interpolate
import random
from statsmodels.stats.multitest import multipletests
import csv

def compute_distance_inter_cluster(data,stim):
    stim_kind = [1,2,3]
    num_tp = np.size(data,0)
    num_dim = np.size(data,2)
    the_mean_data = np.zeros((num_tp,len(stim_kind),num_dim))
    the_distance = np.zeros((math.comb(len(stim_kind),2)+1,num_tp))
    for i in range(len(stim_kind)):
        the_mean_data[:,i,:] = np.mean(data[:,stim==stim_kind[i],:],1)
    for i in range(num_tp):
        the_dis = []
        for j in range(len(stim_kind)-1):
            for k in range(1,len(stim_kind)):
                if j == k:
                    continue
                p1 = the_mean_data[i,j,:]
                p2 = the_mean_data[i,k,:]
                dis = np.sqrt(math.pow(p1[0]-p2[0],2)+math.pow(p1[1]-p2[1],2))
                the_dis.append(dis)
                the_distance[j+k,i] = dis
        the_distance[0,i] = np.mean(the_dis)
    return the_distance

def compute_distance_within_cluster(data,stim):
    stim_kind = [1,2,3]
    num_tp = np.size(data,0)
    the_distance = np.zeros((len(stim_kind)+1,num_tp))
    for i in range(num_tp):
        the_dis = []
        for j in range(len(stim_kind)):
            the_data = np.squeeze(data[i,stim==stim_kind[j],:])
            the_mean = np.mean(the_data,0)
            num_p = np.size(the_data,0)
            if num_p==0:
                continue
            dis = 0
            if the_data.ndim == 1:
                p1 = the_data
            else:
                p1 = the_mean
            for k in range(num_p):
                if the_data.ndim == 1:
                    p2 = the_data
                else:
                    p2 = the_data[k,:]  
                dis = dis+np.sqrt(math.pow(p1[0]-p2[0],2)+math.pow(p1[1]-p2[1],2))
            dis = dis/num_p
            the_dis.append(dis)
            the_distance[j+1,i] = dis
        the_distance[0,i] = np.mean(the_dis)
    return the_distance

def compute_distance_from_origin(data,stim):
    stim_kind = [1,2,3]
    num_tp = np.size(data,0)
    the_distance = np.zeros((len(stim_kind)+1,num_tp))
    p1 = np.mean(np.squeeze(data[0,:,:]),0)
    for i in range(len(stim_kind)):
        the_data = np.mean(data[:,stim==stim_kind[i],:],1)
        for j in range(num_tp):
            a = np.squeeze(the_data[j,:])
            p2 = a
            dis = np.sqrt(math.pow(p1[0]-p2[0],2)+math.pow(p1[1]-p2[1],2))
            the_distance[i+1,j] = dis
    the_distance[0,:] = np.mean(the_distance[1:len(stim_kind)+1,:],0)
    return the_distance

def compute_return_loc(data,stim,tp_range):
    stim_kind = [0,1,2,3]
    num_tp = np.size(data,0)
    num_dim = np.size(data,2)

    # calculate
    the_loc = np.zeros((len(stim_kind),num_dim))
    for i in range(len(stim_kind)):
        if i == 0:
            the_data = np.mean(data,1)
        else:
            the_data = np.mean(data[:,stim==stim_kind[i],:],1)
        the_data = np.mean(the_data[tp_range,:],0)
        the_loc[i,:] = the_data
    return the_loc

def color_map(data, cmap):       
    dmin, dmax = np.nanmin(data), np.nanmax(data)
    cmo = plt.cm.get_cmap(cmap)
    cs, k = list(), 256/cmo.N
    for i in range(cmo.N):
        c = cmo(i)
        for j in range(int(i*k), int((i+1)*k)):
            cs.append(c)
    cs = np.array(cs)
    data = np.uint8(255*(data-dmin)/(dmax-dmin))
    return cs[data]

def compute_return_time(data,return_time_range,ratio):
    num_fly = np.size(data,0)
    return_time_list = np.zeros(num_fly)
    for i in range(num_fly):
        a = np.squeeze(data[i,:])
        max_time = np.argmax(a)
        if np.size(max_time)>1:
            max_time = max_time[0]
        b = a[max_time:]
        return_level = np.mean(a[return_time_range])
        return_time_1 = np.where(b<return_level*(1+ratio))
        return_time_1 = return_time_1[0]
        return_time_list[i] = max_time+return_time_1[0]
    return return_time_list

def compute_return_to_certain_threshold(data,ratio):
    num_fly = np.size(data,0)
    return_time_list = np.zeros(num_fly)
    for i in range(num_fly):
        a = np.squeeze(data[i,:])
        max_time = np.argmax(a)
        if np.size(max_time)>1:
            max_time = max_time[0]
        b = a[max_time:]
        return_level = a[max_time]*ratio
        return_time_1 = np.where(b<return_level)
        if np.size(return_time_1)==0:
            return_time_1 = len(b)-1
            return_time_list[i] = max_time+return_time_1
        else:
            return_time_1 = return_time_1[0]
            return_time_list[i] = max_time+return_time_1[0]
    return return_time_list

def plot_bar(data,color_list,if_save,save_path,x_label,y_label,y_lim,if_paired_test,if_p_corr = False):
    plt.figure(figsize=(2,3))
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if if_paired_test:
        data = np.array(data)
        the_mean = np.mean(data,1)
        the_std = np.std(data,1,ddof = 1)/math.sqrt(np.size(data,1))
        plt.errorbar(range(np.size(data,0)),the_mean,yerr=the_std,ecolor='slategray',elinewidth=1,marker='.',mfc='slategray',\
    mec='slategray',mew=1,ms=1,alpha=1,capsize=5,capthick=3,color='slategray', linewidth=2)
        for j in range(np.size(data,1)):
            plt.plot(range(np.size(data,0)),data[:,j],color = 'slategray',linewidth = 1,alpha = 0.2)
        num_fly = np.size(data,0)
        data.tolist()
    else:
        num_fly = len(data)
        for i in range(num_fly):
            a = data[i]
            the_mean = np.mean(a)
            the_std = np.std(a,ddof = 1)/math.sqrt(len(a))
            # plt.bar([i], the_mean, width=0.7,yerr = the_std,error_kw = {'ecolor' : '0.2', 'capsize' :3 },
            #             alpha=0.7,facecolor = 'white',edgecolor=color_list[i],linewidth=1.5)
            plt.boxplot(a,
                    medianprops={'color': color_list[i], 'linewidth': '1.5'},
                    # meanline=True,
                    # showmeans=True,
                    # meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.5'},
                    showfliers=False,
                    widths  = 0.5,
                    positions= [i],
                    boxprops = {'color': color_list[i], 'linewidth': '1.5'},
                    whiskerprops = {'color': color_list[i], 'linewidth': '1.5'},
                    capprops =  {'color': color_list[i], 'linewidth': '1.5'},
                    flierprops={"marker": "o", "markerfacecolor": "black","markeredgecolor":"black", "markersize": 2})
            jittered_x = np.random.normal(i, 0.05, len(a)) 
            plt.scatter(jittered_x, a, color='black', alpha=0.3,s = 10)
    p_list = []
    for i in range(num_fly-1):
        for j in range(i+1,num_fly):
            a = data[i]
            b = data[j]
            if if_paired_test:
                res = wilcoxon(a,b)
            else:
                res = mannwhitneyu(a,b)
            p = res.pvalue
            p_list.append(p)
    if if_p_corr:
        p_corrected = multipletests(p_list,method = 'fdr_bh')
        p_list = p_corrected[1]
    for i in range(len(p_list)):
        p = p_list[i]
        if p<0.05 and p>=0.01:
            plt.text(0.5*(i+1),np.mean(data[0]),'*',verticalalignment = 'center', horizontalalignment = 'center')
        elif p<0.01 and p>0.001:
            plt.text(0.5*(i+1),np.mean(data[0]),'**',verticalalignment = 'center', horizontalalignment = 'center')
        elif p<0.001 and p>=0.0001:
            plt.text(0.5*(i+1),np.mean(data[0]),'***',verticalalignment = 'center', horizontalalignment = 'center') 
        elif p<0.0001:
            plt.text(0.5*(i+1),np.mean(data[0]),'****',verticalalignment = 'center', horizontalalignment = 'center') 
    p_result = []
    p_result.append(p_list)
    plt.xticks([0,1,2],x_label)
    plt.ylabel(y_label)
    plt.ylim(y_lim)
    if np.size(save_path)>0 and if_save:
        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42
        plt.savefig(save_path+'.pdf',bbox_inches = 'tight')
        plt.savefig(save_path+'.png',bbox_inches = 'tight')
        with open(save_path+'-p.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(p_result)
    plt.show()