import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
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

def normalization(data, new_min, new_max):
    old_min = np.min(data)
    old_max = np.max(data)
    return (data - old_min) * (new_max - new_min) / (old_max - old_min) + new_min


region_names = {64: 'MBPED',65: 'MBVL',66: 'MBML',55: 'LH',72: 'SLP',73: 'SIP',74: 'SMP',63: 'CRE',84: 'SCL',59: 'ICL',
                4: 'NO',23: 'EB',26: 'FB',56: 'LAL',79: 'AOTU',75: 'AVLP',76: 'PVLP',77: 'IVLP',60: 'VES',80: 'GOR',82: 'SPS',
                85: 'EPA',67: 'FLA'}
data_index_list = ['20230417-fly2', '20230420-fly2', '20230420-fly3', '20230428-fly1', '20230507-fly1', 
                   '20230510-fly1', '20230510-fly2', '20230511-fly2', '20230511-fly3', '20230515-fly1']
focus_regions_0 = [64,65,66,55,72,73,74,63,84,59,4,23,26,56,79,75,76,77,60,80,82,85,67] # all brain regions
focus_regions_1 = [64,65,66,55,72,73,74,63,84,59,4,23,26,56,79,75,76,77,60,85,67] # brain regions except 80,82,
focus_regions_2 = [64,65,66,72,73,74,63,84,59,4,23,26,56,79,75,76,77,60,85,67] # brain regions except 55,80,82,

# logger
log_file_name = 'motion-corr-g7f-ach'
logging_name = 'motion-corr'
today = datetime.datetime.now()
formatted_date = today.strftime('%Y-%m-%d')
logger = logger_config(log_path = 'log_' + log_file_name + "_" + formatted_date + '.log', logging_name = logging_name)
logger.info('Start the code ...')


''' correlation '''

## path
path_inspur = '/as13000_sam2p/0-LLB/olfactory_representation/'
name = "g7f-ach"

delay_list = list(np.arange(-20,22,2))
logger.info("delay_list:{}".format(delay_list))

for data_index in data_index_list: 
    logger.info('============ data_index ' + data_index + ' ===========')
    
    path_output = path_inspur + 'Ach-' + data_index + '-pre_stimu/'
    
    focus_regions = []
    if data_index in ['20230428-fly1', '20230510-fly1']:
        focus_regions = focus_regions_1
    elif data_index == '20230507-fly1':
        focus_regions = focus_regions_2
    else:
        focus_regions = focus_regions_0
        
    k = 10
    motion_strength_down = np.load(path_inspur + "motion_res/motion_strength_down_807_up10_" + data_index + ".npy")
    logger.info("focus_regions:{}, motion_strength_down:{}".format(len(focus_regions), motion_strength_down.shape))
    n_time = len(motion_strength_down)

    regions_corr, regions_p, region_cnt = [], [], []
    for region_idx in focus_regions:
        ## load data
        records_region = np.load(path_output +'neuron_concat_records/region_' + str(region_idx) + '_neuron_records.npy',allow_pickle=True).tolist()
        region_cnt.append(len(records_region))
        
        corrs, ps = [], []
        for rec in records_region:
            rec = np.interp(np.arange(0, len(rec), 1 / k), np.arange(0, len(rec)), rec)

            delay_corr, corr_p = [], []
            for delay in delay_list:
                pearson = pearsonr(motion_strength_down[25 : n_time - 25], rec[25 + delay : n_time - 25 + delay])
                delay_corr.append(pearson[0])
                if np.isnan(pearson[1]):
                    corr_p.append(100)
                else:
                    corr_p.append(pearson[1])
            corrs.append(delay_corr)
            ps.append(corr_p)
        
        regions_corr.extend(corrs)
        regions_p.extend(ps)
    logger.info("regions_corr:{}, regions_p:{}, region_cnt:{}".format(np.array(regions_corr).shape, np.array(regions_p).shape, len(region_cnt)))
    
    np.save(path_inspur + "motion_res/corr/motion_calc_corr_delay_r_" + name + "_" + data_index + ".npy", regions_corr)
    np.save(path_inspur + "motion_res/corr/motion_calc_corr_delay_p_" + name + "_" + data_index + ".npy", regions_p)
    np.save(path_inspur + "motion_res/corr/motion_calc_corr_delay_neuron_cnt_" + name + "_" + data_index + ".npy", region_cnt)

    figure = plt.figure(figsize=(3,8)) 
    axes = figure.add_subplot(111) 
    caxes = axes.matshow(regions_corr, interpolation ='nearest', cmap = "Purples", aspect = 'auto') 
    plt.yticks([])
    plt.title("region correlation", fontsize = 16)
    plt.xlabel("regions", fontsize = 14)
    figure.colorbar(caxes) 
    plt.savefig(path_inspur + "motion_res/corr/motion_calc_corr_delay_r_" + name + "_" + data_index + ".png", bbox_inches='tight')
    plt.show()

logger.info("END")
    
