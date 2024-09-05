import numpy as np
import os
from skimage import io
from utils.projection_3d import *
from matplotlib.colors import LinearSegmentedColormap

file_path = 'Z:/0-FJQ/olfactory_representation/pipeline-revision/results/nsyb-G7f-rAch1h/figures-for-revision1/ensemble_significance/rAch/summary-2/tif'
file_prefix = 'Transform_fly_'

fly_list = [1,5,6,8,9]
colors = ["white", '#751C77']
n_bins = 100 
cmap_name = 'white_to_blue'
white_to_blue = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

# fly_list = [0,1,2,3,4,5,6,7,8,9]
# colors = ["white", '#036EB8']
# n_bins = 100 
# cmap_name = 'white_to_blue'
# white_to_blue = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)


channel_list = [1,2]
odor_list = [1,2,3]
cut_z = [28,130] # z-axis cut range
print(cut_z)
pixel_size = [318.51/512,318.51/512 ,218/218] ## pixel size

savepath = file_path+'/avg'
if not os.path.exists(savepath):
    os.mkdir(savepath)

for i in channel_list:
    for j in odor_list:
        print(str(i)+' '+str(j))
        result = []
        for m in fly_list:
            weight = io.imread(file_path+'/'+file_prefix+str(m)+'_channel_'+str(i)+'_odor_'+str(j)+'.tif')
            weight[weight==0] = 15000
            weight = weight/10000*2-3
            result.append(weight)
        result = np.mean(np.stack(result),0)
        io.imsave(savepath + '/channel_'+str(i)+'_odor_'+str(j)+'.tif', ((result+3)/2*10000).astype('uint16'))
        ## cut z-axis
        result = result[cut_z[0]:cut_z[1],:,:]
        ## plot 3d image 
        projection_3d(result.transpose((1,2,0)),512*pixel_size[0],512*pixel_size[1],result.shape[0]*pixel_size[2],30,
                      white_to_blue,[0.05,0.2],0,True,True,os.path.join(savepath, 'channel_'+str(i)+'_odor_'+str(j)+'.'))      