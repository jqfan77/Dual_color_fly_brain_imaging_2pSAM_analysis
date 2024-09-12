import tifffile as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import datetime

def logger_config(log_path, logging_name):
    logger = logging.getLogger(logging_name)
    logger.setLevel(level = logging.DEBUG)
    handler = logging.FileHandler(log_path, encoding = 'UTF-8')
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    logger.addHandler(handler)

    return logger

data_index_list = ['20230417-fly2', '20230420-fly2', '20230420-fly3', '20230428-fly1', '20230507-fly1', '20230510-fly1', '20230510-fly2', '20230511-fly2', '20230511-fly3', '20230515-fly1']
focus_regions_0 = [64,65,66,55,72,73,74,63,84,59,4,23,26,56,79,75,76,77,60,80,82,85,67] # all brain regions
focus_regions_1 = [64,65,66,55,72,73,74,63,84,59,4,23,26,56,79,75,76,77,60,85,67] # brain regions except 80,82,
focus_regions_2 = [64,65,66,72,73,74,63,84,59,4,23,26,56,79,75,76,77,60,85,67] # brain regions except 55,80,82,

# logger
log_file_name = 'concate-stimu-g7f-for-ach-flies'
logging_name = 'concate-stimu'
today = datetime.datetime.now()
formatted_date = today.strftime('%Y-%m-%d')
logger = logger_config(log_path = 'log_' + log_file_name + "_" + formatted_date + '.log', logging_name = logging_name)
logger.info('Start the code ...')


for data_index in data_index_list: 
    logger.info('============ data_index ' + data_index + ' ===========')

    # 01 settiings

    ## parameters
    date_id = data_index.split('-')[0]
    fly_id = data_index.split('-')[1]
    n_trails = 180

    ## path
    path = '/as13000_sam2p/0-FJQ/olfactory_representation/data/' + date_id + '-nsyb-G7f-rAch1h/' + fly_id + '/data/'
    path_atlas  = '/as13000_sam2p/0-FJQ/olfactory_representation/data/' + date_id + '-nsyb-G7f-rAch1h/' + fly_id + '/align_to_atlas/'
    path_output = '/as13000_sam2p/0-LLB/olfactory_representation/Ach-' + data_index + '-stimu/'


    # 02 Load data

    ## load G7f data
    dff0_con_C2 = np.load(path + 'dff0_-3-20_down1_C2.npy') 
    logger.info(type(dff0_con_C2), dff0_con_C2.shape)

    ## load atlas data
    image_file = path_atlas + 'Transformed_atlas_eroded_r5.tif'
    img_tf = tf.imread(image_file)

    ## list the brain region indices
    region_idx = []
    for i in range(13,38): # only need the 13th - 38th layers of atlas
        non_val = img_tf[i].ravel()[np.flatnonzero(img_tf[i])].tolist()
        region_idx += list(set(non_val))
        ## plot atlas
        data = pd.DataFrame(img_tf[i])
        sns.heatmap(data)
        plt.show()
    region_idx = list(set(region_idx))
    logger.info("# region indices: ", len(region_idx), " region indices: ", region_idx)


    # 03 Concatenate G7f recordings

    ## Splicing multiple sessions for each neuron
    for region_idx in focus_regions_0:
        ## save the neuron positions of each region
        neurons_positions = {}    
        for i in range(13,38):
            x_idx, y_idx = np.where(img_tf[i] == region_idx)
            if len(x_idx) > 0 and len(y_idx) > 0:
                neurons_positions[i] = [(x_idx[j], y_idx[j]) for j in range(len(x_idx))]
        np.save(path_output + 'neuron_concat_records/region_' + str(region_idx) + '_neuron_positions.npy', neurons_positions)
        logger.info('region_' + str(region_idx) + ' save neuron positions!')
        
        ## save the neuron G7f records of each region
        neuron_records = []
        for k,v in neurons_positions.items():
            z = k - 13
            for x_y in v:
                neuron_record = []
                for i in range(n_trails):                
                    neuron_record += list(dff0_con_C2[i, :, z, x_y[0], x_y[1]]) 
                neuron_records.append(neuron_record)
        np.save(path_output + 'neuron_concat_records/region_' + str(region_idx) + '_neuron_records.npy', neuron_records)
        if len(neuron_records) == 0:
            logger.info('neuron records: ', 0, 0)
        else:
            logger.info('neuron records: ', len(neuron_records),len(neuron_records[0]))
        
        del neuron_records 
        del neurons_positions

    del dff0_con_C2
logger.info('END')
