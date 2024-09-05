import numpy as np
import os
from skimage import io
from utils.projection_3d import *
from matplotlib.colors import LinearSegmentedColormap

colors = ["white", '#006934']
n_bins = 100  
cmap_name = 'white_to_blue'
white_to_blue = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

file_path_1 = 'Z:/0-FJQ/olfactory_representation/pipeline-revision/results/nsyb-G7f-rAch1h/figures-for-revision1/ensemble_significance/rAch/summary-2/tif/avg'
file_path_2 = 'Z:/0-FJQ/olfactory_representation/pipeline-revision/results/nsyb-G7f-r5HT1.0/figures-for-revision1/ensemble_significance/r5HT/summary-2/tif/avg'
channel_list = [1]
odor_list = [1,2,3]
cut_z = [28,130] # z-axis cut range
print(cut_z)
pixel_size = [318.51/512,318.51/512 ,218/218] ## pixel size

savepath = 'Z:/0-FJQ/olfactory_representation/pipeline-revision/results/nsyb-G7f-rAch1h/figures-for-revision1/ensemble_significance/G7f_avg'
if not os.path.exists(savepath):
    os.mkdir(savepath)

for i in channel_list:
    for j in odor_list:
        print(str(i)+' '+str(j))
        result = []
        weight_1 = io.imread(file_path_1+'/'+'channel_'+str(i)+'_odor_'+str(j)+'.tif')
        weight_1[weight_1==0] = 15000
        weight_1 = weight_1/10000*2-3
        weight_2 = io.imread(file_path_2+'/'+'channel_'+str(i)+'_odor_'+str(j)+'.tif')
        weight_2[weight_2==0] = 15000
        weight_2 = weight_2/10000*2-3
        weight = (weight_1+weight_2)/2
        ## cut z-axis
        result = weight[cut_z[0]:cut_z[1],:,:]
        ## plot 3d image 
        projection_3d(result.transpose((1,2,0)),512*pixel_size[0],512*pixel_size[1],result.shape[0]*pixel_size[2],30,
                      white_to_blue,[0.03,0.18],0,True,True,os.path.join(savepath, 'channel_'+str(i)+'_odor_'+str(j)+'.'))      