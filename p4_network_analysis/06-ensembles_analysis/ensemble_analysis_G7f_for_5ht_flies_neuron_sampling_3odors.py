import numpy as np
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

region_names = {64: 'MBPED',65: 'MBVL',66: 'MBML',55: 'LH',72: 'SLP',73: 'SIP',74: 'SMP',63: 'CRE',84: 'SCL',59: 'ICL',
                4: 'NO',23: 'EB',26: 'FB',56: 'LAL',79: 'AOTU',75: 'AVLP',76: 'PVLP',77: 'IVLP',60: 'VES',80: 'GOR',82: 'SPS',
                85: 'EPA',67: 'FLA'}
data_index_list = ['20230429-r5HT1.0-fly1', '20230506-r5HT1.0-fly1', '20230513-r5HT1.0-fly1', 
                    '20230513-r5HT1.0-fly2', '20230516-r5HT1.0-fly2', '20230516-r5HT1.0-fly4',
                    '20230517-r5HT1.0-fly1', '20230601-r5HT1.0-fly1', '20230601-r5HT1.0-fly3', '20230603-r5HT1.0-fly1']
focus_regions_0 = [64,65,66,55,72,73,74,63,84,59,4,23,26,56,79,75,76,77,60,80,82,85,67]
focus_regions_1 = [64,65,66,55,72,73,74,63,84,59,4,23,26,56,79,75,76,77,60,80,85,67] # 82
focus_regions_2 = [64,65,66,55,72,73,74,63,84,59,4,23,26,56,79,75,76,77,60,85,67] # 80,82
focus_regions_3 = [64,65,66,72,73,74,63,84,59,4,23,26,56,79,75,76,77,60,80,82,85,67] # 55

# logger
log_file_name = 'ensemble-sampling-g7f-5ht'
logging_name = 'ensemble-sampling'
today = datetime.datetime.now()
formatted_date = today.strftime('%Y-%m-%d')
logger = logger_config(log_path = 'log_' + log_file_name + "_" + formatted_date + '.log', logging_name = logging_name)
logger.info('Start the code ...')


''' match neurons by tuning '''

## path
path_inspur = '/sam2p/0-LLB/olfactory_representation/'
path_source = "/sam2p/0-FJQ/olfactory_representation/data/"

name = "5ht"
n_trails = 180

for fly_cnt in range(len(data_index_list)): 
    data_index = data_index_list[fly_cnt]
    logger.info('============ data_index ' + data_index + ' == fly_cnt' + str(fly_cnt) + '==============')
    
    path_output = path_inspur + data_index + '-trans-stimu/'
    date_id = data_index.split('-')[0]
    fly_id = data_index.split('-')[2]
    path = path_source + date_id + '-nsyb-G7f-r5HT1.0/' + fly_id + '/data/'
    chan = 1
    dff0_con_C = np.load(path + 'dff0_-3-20_down1_C2.npy') 
    logger.info("dff0_con_C {}, {}".format(type(dff0_con_C), dff0_con_C.shape))
    
    logger.info('---------- chan  {} --------'.format(chan))
    voxel_select_dict = np.load(path_inspur +'ensemble_res/' + name + '_sample_neuron_positions_dict_tuning_fly_' + str(fly_cnt) + '_channel_' + str(chan) + '.npy', allow_pickle = True).item()

    ## method 2
    neuron_positions = []
    neuron_records = []
    for k,v in voxel_select_dict.items():
        for x_y in v:
            neuron_positions.append((x_y[0], x_y[1], k))
            neuron_record = []
            for i in range(n_trails):                
                neuron_record += list(dff0_con_C[i, :, k, x_y[0], x_y[1]]) 
            neuron_records.append(neuron_record)
    logger.info("neuron_positions:{},{}".format(len(neuron_positions), neuron_positions[:5]))
    logger.info("neuron_records:{},{}".format(len(neuron_records), neuron_records[:5]))

    np.save(path_inspur +'ensemble_res/' + name + '_sample_neuron_positions_list_tuning_fly_' + str(fly_cnt) + '_channel_' + str(chan) + '.npy', neuron_positions)
    np.save(path_inspur +'ensemble_res/' + name + '_sample_neuron_records_tuning_fly_' + str(fly_cnt) + '_channel_' + str(chan) + '.npy', neuron_records)
    logger.info("finish saving positions and records...")

logger.info("END")
    
