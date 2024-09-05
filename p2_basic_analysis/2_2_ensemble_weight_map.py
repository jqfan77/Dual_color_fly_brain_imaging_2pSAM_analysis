import numpy as np
import os
from skimage import io
from utils.projection_3d import *
from matplotlib.colors import LinearSegmentedColormap
colors = ["white", '#006934']
n_bins = 100 
cmap_name = 'white_to_green'
white_to_green = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

colors = ["white", '#751C77']
n_bins = 100  
cmap_name = 'white_to_purple'
white_to_purple = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

cmap_list = [white_to_green,white_to_green,white_to_purple,white_to_purple]

file_path = 'Z:/0-FJQ/results/nsyb-G7f-rAch1h-ensemble_weight/summary_2/tif_new'
file_prefix = 'Transform_fly_'
fly_list = range(10)
channel_list = [2]
odor_list = [0,1,2]
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
            weight[weight==0] = 10000
            weight = weight/10000-1
            result.append(weight)
        result = np.mean(np.stack(result),0)
        io.imsave(savepath + '/channel_'+str(i)+'_odor_'+str(j)+'.tif', ((result+1)*10000).astype('uint16'))
        ## cut z-axis
        result = result[cut_z[0]:cut_z[1],:,:]
        ## plot 3d image 
        if i==1:
            projection_3d(result.transpose((1,2,0)),512*pixel_size[0],512*pixel_size[1],result.shape[0]*pixel_size[2],30,
                        cmap_list[i],[0,0.002],0,True,True,os.path.join(savepath, 'channel_'+str(i)+'_odor_'+str(j)+'.'))      
        else:
            if j==0:
                projection_3d(result.transpose((1,2,0)),512*pixel_size[0],512*pixel_size[1],result.shape[0]*pixel_size[2],30,
                        cmap_list[i],[0,0.004],0,True,True,os.path.join(savepath, 'channel_'+str(i)+'_odor_'+str(j)+'.')) 
            else:
                projection_3d(result.transpose((1,2,0)),512*pixel_size[0],512*pixel_size[1],result.shape[0]*pixel_size[2],30,
                        cmap_list[i],[0,0.006],0,True,True,os.path.join(savepath, 'channel_'+str(i)+'_odor_'+str(j)+'.')) 