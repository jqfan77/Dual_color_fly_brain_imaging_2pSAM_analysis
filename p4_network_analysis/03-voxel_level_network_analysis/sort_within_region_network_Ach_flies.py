import numpy as np
import matplotlib.pyplot as plt
import logging
import datetime
import copy

def logger_config(log_path, logging_name):
    logger = logging.getLogger(logging_name)
    logger.setLevel(level = logging.DEBUG)
    handler = logging.FileHandler(log_path, encoding = 'UTF-8')
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    logger.addHandler(handler)

    return logger

## define normalization method
def normalization(data, new_min, new_max):
    old_min = np.min(data)
    old_max = np.max(data)

    return (data - old_min) * (new_max - new_min) / (old_max - old_min) + new_min


data_index_list = ['20230417-fly2', '20230420-fly2', '20230420-fly3', '20230428-fly1', '20230507-fly1', '20230510-fly1', '20230510-fly2', '20230511-fly2', '20230511-fly3', '20230515-fly1']
focus_regions_0 = [64,65,66,55,72,73,74,63,84,59,4,23,26,56,79,75,76,77,60,80,82,85,67] # all brain regions
focus_regions_1 = [64,65,66,55,72,73,74,63,84,59,4,23,26,56,79,75,76,77,60,85,67] # brain regions except 80,82,
focus_regions_2 = [64,65,66,72,73,74,63,84,59,4,23,26,56,79,75,76,77,60,85,67] # brain regions except 55,80,82,

# logger
log_file_name = 'region-network-ach-flies-sort'
logging_name = 'region-network'
today = datetime.datetime.now()
formatted_date = today.strftime('%Y-%m-%d')
logger = logger_config(log_path = 'log_' + log_file_name + "_" + formatted_date + '.log', logging_name = logging_name)
logger.info('Start the code ...')

path_res = '/sam2p/0-LLB/olfactory_representation/single_region_res/'

for data_index in data_index_list: 
    logger.info('============ data_index ' + data_index + ' ===========')

    ## parameters
    date_id = data_index.split('-')[0]
    fly_id = data_index.split('-')[1]
    n_trails = 180

    focus_regions = []
    ## Some brain regions of the specified flies are not used
    if data_index in ['20230428-fly1', '20230510-fly1']:
        focus_regions = focus_regions_1
    elif data_index == '20230507-fly1':
        focus_regions = focus_regions_2
    else:
        focus_regions = focus_regions_0

    ## generate network for each region
    for region_idx in [66,55,64,65,73,63,84,59,23,26,85,79,77]:   
        logger.info("---- processing region " + str(region_idx) + " ----")
        
        if region_idx not in focus_regions:
            continue

        ## plot G7f correlation matrix
        path_output = '/sam2p/0-LLB/olfactory_representation/Ach-' + data_index + '-stimu/'
        logger.info("path_output:{}".format(path_output))
        calc_hierarchy_ids = np.load(path_output + 'network/region_' + str(region_idx) +'_all_neurons_similarity_hierarchy_ids.npy',allow_pickle=True).tolist()
        # load record data
        similarity = np.load(path_output + 'network/region_' + str(region_idx) +'_all_neurons_similarity_hierarchy.npy',allow_pickle=True)
        figure = plt.figure(figsize=(8,8)) 
        axes = figure.add_subplot(111) 
        caxes = axes.matshow(similarity, interpolation ='nearest', vmin = -0.2, vmax = 1) 
        figure.colorbar(caxes) 
        plt.savefig(path_res + 'region_' + str(region_idx) +'_all_neurons_similarity_hierarchy_' + data_index + '.png',  dpi = 300, bbox_inches='tight')
        plt.savefig(path_res + 'region_' + str(region_idx) +'_all_neurons_similarity_hierarchy_' + data_index + '.pdf',  dpi = 300, bbox_inches='tight')
        plt.show()
        A1 = copy.deepcopy(similarity)
        logger.info("-- finish ploting calc similarity:{}, calc_hierarchy_ids:{}".format(similarity.shape, len(calc_hierarchy_ids)))

        ## plot ach correlation matrix
        path_output = '/sam2p/0-LLB/olfactory_representation/Ach-' + data_index + '-trans-stimu/'
        logger.info("path_output:{}".format(path_output))
        ach_hierarchy_ids = np.load(path_output + 'network/region_' + str(region_idx) +'_all_neurons_similarity_hierarchy_ids.npy',allow_pickle=True).tolist()
        # load record data
        ori_similarity = np.load(path_output + 'network/region_' + str(region_idx) +'_all_neurons_similarity_hierarchy.npy',allow_pickle=True)
        n_neurons = len(ach_hierarchy_ids)
        similarity = np.zeros((n_neurons, n_neurons))
        for i in range(0, n_neurons - 1):
            for j in range(i + 1, n_neurons):
                x0, y0 = calc_hierarchy_ids[i], calc_hierarchy_ids[j]
                x, y = ach_hierarchy_ids.index(x0), ach_hierarchy_ids.index(y0)
                similarity[i,j] = similarity[j,i] = ori_similarity[x,y]
        logger.info("similarity:{}".format(similarity.shape))
        figure = plt.figure(figsize=(8,8)) 
        axes = figure.add_subplot(111) 
        caxes = axes.matshow(similarity, interpolation ='nearest', vmin = -0.2, vmax = 1) 
        figure.colorbar(caxes) 
        plt.savefig(path_res + 'region_' + str(region_idx) +'_all_neurons_similarity_hierarchy1_' + data_index + '.png',  dpi = 300, bbox_inches='tight')
        plt.savefig(path_res + 'region_' + str(region_idx) +'_all_neurons_similarity_hierarchy1_' + data_index + '.pdf',  dpi = 300, bbox_inches='tight')
        plt.show()
        A2 = copy.deepcopy(similarity)
        np.save(path_output + 'network/region_' + str(region_idx) +'_all_neurons_similarity_hierarchy_sorted_as_calc.npy', similarity)
        logger.info("-- finish ploting 5ht similarity:{}, hierarchy_ids:{}".format(similarity.shape, len(ach_hierarchy_ids)))

        if len(A1) != len(A2):
            logger.info("ERROR: len(A1) != len(A2)")

        ## plot Delta correlation matrix
        A1 = np.nan_to_num(A1)
        A1_avg = np.full((len(A1), len(A1)), np.mean(A1))
        A1_delta = A1 - A1_avg
        A1_delta = A1_delta / A1_avg
        
        A2 = np.nan_to_num(A2)
        A2_avg = np.full((len(A2), len(A2)), np.mean(A2))
        A2_delta = A2 - A2_avg
        A2_delta = A2_delta / A2_avg

        A = A2_delta - A1_delta
        A = np.nan_to_num(A)
        np.save(path_output + 'network/region_' + str(region_idx) +'_all_neurons_similarity_hierarchy_dff_sorted_as_calc.npy', A)
        logger.info("finish saving delta similarity...")

        A = normalization(A, -1, 1)
        figure = plt.figure(figsize=(8,8)) 
        axes = figure.add_subplot(111) 
        caxes = axes.matshow(A, interpolation ='nearest', cmap = "coolwarm", vmin = -1, vmax = 1) 
        figure.colorbar(caxes) 
        plt.savefig(path_res + 'region_' + str(region_idx) +'_all_neurons_similarity_hierarchy_dff_' + data_index + '.png',  dpi = 300, bbox_inches='tight')
        plt.savefig(path_res + 'region_' + str(region_idx) +'_all_neurons_similarity_hierarchy_dff_' + data_index + '.pdf',  dpi = 300, bbox_inches='tight')
        plt.show()
        logger.info("-- finish ploting delta similarity:{}".format(A.shape))
        
logger.info("END")
