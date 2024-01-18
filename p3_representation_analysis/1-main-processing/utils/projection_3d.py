def projection_3d(data,x_scale,y_scale,z_scale,space,colormapname,caxis,projection_type,ifcolorbar,ifsave,savepath):
    # projection_type: 0-max;1-mean
    import numpy as np
    import matplotlib.pyplot as plt
    if projection_type == 0:
        data_px = np.max(data,0)
        data_py = np.max(data,1)
        data_pz = np.max(data,2)
    else:
        data_px = np.mean(data,0)
        data_py = np.mean(data,1)
        data_pz = np.mean(data,2)
    # plot
    x_scale = int(x_scale)
    y_scale = int(y_scale)
    z_scale = int(z_scale)
    plt.figure(figsize=(6,6))
    grid = plt.GridSpec(y_scale + z_scale + space, x_scale + z_scale + space)
    plt.subplot(grid[0:y_scale,0:x_scale])
    plt.imshow(data_pz,cmap = colormapname,vmin = caxis[0],vmax = caxis[1],aspect = 'auto')
    plt.xticks([]),plt.yticks([])
    plt.subplot(grid[0:y_scale,x_scale+space:x_scale+z_scale+space]) # y
    plt.imshow(data_py[:,::-1],cmap = colormapname,vmin = caxis[0],vmax = caxis[1],aspect = 'auto')
    plt.xticks([]),plt.yticks([])
    plt.subplot(grid[y_scale+space:y_scale+z_scale+space,0:x_scale]) # x
    plt.imshow(np.transpose(data_px[:,::-1]),cmap = colormapname,vmin = caxis[0],vmax = caxis[1],aspect = 'auto')
    plt.xticks([]),plt.yticks([])
    if ifcolorbar:
        plt.subplot(grid[y_scale+space:y_scale+z_scale+space,x_scale+space:x_scale+z_scale+space]) 
        plt.imshow(data_py,cmap = colormapname,vmin = caxis[0],vmax = caxis[1],aspect = 'auto')
        plt.xticks([]),plt.yticks([])
        plt.colorbar()
    if ifsave:
        plt.savefig(savepath, dpi = 300)
    plt.show()
    plt.close()
