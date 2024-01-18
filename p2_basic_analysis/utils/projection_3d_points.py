import ipdb

def projection_3d_points(data,point,x_scale,y_scale,z_scale,space,colormapname,caxis,projection_type,ifcolorbar,ifsave,savepath):
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
   
    point_xy = np.max(point,0)
    point_xy_1 = np.where(point_xy != 0)
    point_xz = np.max(point,2)
    point_xz_1 = np.where(point_xz != 0)
    point_yz = np.max(point,1)
    point_yz_1 = np.where(point_yz != 0)

    # plot
    x_scale = int(x_scale)
    y_scale = int(y_scale)
    z_scale = int(z_scale)
    plt.figure(figsize=(6,6))
    grid = plt.GridSpec(y_scale + z_scale + space, x_scale + z_scale + space)
    plt.subplot(grid[0:y_scale,0:x_scale])
    plt.imshow(data_pz,cmap = colormapname,vmin = caxis[0],vmax = caxis[1],aspect = 'auto')
    plt.xticks([]),plt.yticks([])
    plt.scatter(point_xy_1[1],point_xy_1[0],s=1,c='m',marker='*')

    plt.subplot(grid[0:y_scale,x_scale+space:x_scale+z_scale+space]) # 共y轴
    plt.imshow(data_py[:,::-1],cmap = colormapname,vmin = caxis[0],vmax = caxis[1],aspect = 'auto')
    plt.xticks([]),plt.yticks([])
    plt.scatter(point.shape[0] - point_xz_1[0] - 1,point_xz_1[1],s=1,c='m',marker='*')


    plt.subplot(grid[y_scale+space:y_scale+z_scale+space,0:x_scale]) # 共x轴
    plt.imshow(np.transpose(data_px[:,::-1]),cmap = colormapname,vmin = caxis[0],vmax = caxis[1],aspect = 'auto')
    plt.xticks([]),plt.yticks([])
    plt.scatter(point_yz_1[1],point.shape[0] - point_yz_1[0] - 1,s=1,c='m',marker='*')

    if ifcolorbar:
        plt.subplot(grid[y_scale+space:y_scale+z_scale+space,x_scale+space:x_scale+z_scale+space]) 
        plt.imshow(data_py,cmap = colormapname,vmin = caxis[0],vmax = caxis[1],aspect = 'auto')
        plt.xticks([]),plt.yticks([])
        plt.colorbar()
    if ifsave:
        plt.savefig(savepath+'png', dpi = 300)
        plt.savefig(savepath+'eps', dpi = 300)
    # plt.show()
    plt.close()
