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

region_names = {64: 'MBPED',65: 'MBVL',66: 'MBML',55: 'LH',72: 'SLP',73: 'SIP',74: 'SMP',63: 'CRE',84: 'SCL',59: 'ICL',
                4: 'NO',23: 'EB',26: 'FB',56: 'LAL',79: 'AOTU',75: 'AVLP',76: 'PVLP',77: 'IVLP',60: 'VES',80: 'GOR',82: 'SPS',
                85: 'EPA',67: 'FLA'}
data_index_list = ['20230417-fly2', '20230420-fly2', '20230420-fly3', '20230428-fly1', '20230507-fly1', '20230510-fly1', '20230510-fly2', '20230511-fly2', '20230511-fly3', '20230515-fly1']
focus_regions_0 = [64,65,66,55,72,73,74,63,84,59,4,23,26,56,79,75,76,77,60,80,82,85,67] # all brain regions
focus_regions_1 = [64,65,66,55,72,73,74,63,84,59,4,23,26,56,79,75,76,77,60,85,67] # brain regions except 80,82,
focus_regions_2 = [64,65,66,72,73,74,63,84,59,4,23,26,56,79,75,76,77,60,85,67] # brain regions except 55,80,82,

stim = np.array([3,1,2,3,2,1,3,1,2,1,2,3,1,2,3,2,3,1,3,2,1,2,3,1,2,3,1,2,1,3,2,3,1,2,3,1,3,1,2,3,1,2,3,2,1,2,1,3,2,1,3,1,2,3,\
                 1,2,3,2,1,3,1,2,3,2,1,3,1,3,2,3,2,1,3,2,1,3,1,2,3,2,1,3,1,2,3,2,1,2,3,1,3,1,2,3,1,2,3,2,1,2,3,1,2,1,3,2,1,3,\
                 1,3,2,3,1,2,1,2,3,2,3,1,2,3,1,3,2,1,2,3,1,2,1,3,1,2,3,2,3,1,2,1,3,1,3,2,3,1,2,1,2,3,2,1,3,1,2,3,2,3,1,3,1,2,\
                 1,3,2,1,3,2,3,1,2,3,2,1,2,1,3,1,2,3])

# logger
log_file_name = 'ensemble-network-g7f-ach-tuning'
logging_name = 'ensemble-network'
today = datetime.datetime.now()
formatted_date = today.strftime('%Y-%m-%d')
logger = logger_config(log_path = 'log_' + log_file_name + "_" + formatted_date + '.log', logging_name = logging_name)
logger.info('Start the code ...')


''' odor correlation matrix '''

## path
path_inspur = '/as13000_sam2p/0-LLB/olfactory_representation/'
path_source = "/as13000_sam2p/0-FJQ/olfactory_representation/data/"

for data_index in data_index_list: 
    fly_cnt = data_index_list.index(data_index)
    logger.info('============ data_index ' + data_index + ' ===========')

    # 01 settiings

    ## parameters
    date_id = data_index.split('-')[0]
    fly_id = data_index.split('-')[1]


    ## path
    path = path_source + date_id + '-nsyb-G7f-rAch1h/' + fly_id + '/data/'
    path_atlas  = path_source + date_id + '-nsyb-G7f-rAch1h/' + fly_id + '/align_to_atlas/'
    path_output = path_inspur + 'Ach-' + data_index + '-stimu/'

    ''' 3-odors '''
    name = 'ach'
    chan = 1
    records_sample = np.load(path_inspur + 'ensemble_res/' + name + '_sample_neuron_records_tuning_fly_' + str(fly_cnt) + '_channel_' + str(chan) + '.npy',allow_pickle=True)
    logger.info("records_sample: {}, {}".format(type(records_sample), records_sample.shape))

    ## trial avg    
    records_avg = []
    for odor in range(3):
        odor_stim = [i for i in range(len(stim)) if stim[i] == odor + 1]
        frames, duration = 20, 23
        select_data = records_sample[:, 3 + odor_stim[0] * duration : (1 + odor_stim[0]) * duration]
        cnt = 1
        while cnt < len(odor_stim):
            start_timepoint = 3 + odor_stim[cnt] * duration
            select_data += records_sample[:, start_timepoint : start_timepoint + frames] 
            cnt += 1
        select_data = select_data / len(odor_stim)
        logger.info("odor: {} -- select_data: {}".format(odor, select_data.shape))
        if odor == 0:
            records_avg = select_data
        else:
            records_avg = np.concatenate((records_avg, select_data), axis = 1) 
    logger.info("Concate records_avg: {}".format(np.array(records_avg).shape))

    ## calculate pearson correlation
    n_neurons = len(records_sample)
    similarity = np.zeros((n_neurons, n_neurons))
    for i in range(n_neurons - 1):
        for j in range(i + 1, n_neurons):   
            pearson = pearsonr(records_avg[i], records_avg[j])
            similarity[i,j] = similarity[j,i] = pearson[0] if not np.isnan(pearson[0]) else 0
    np.save(path_output + 'network/sample_neuron_similarity_3odors_tuning_trialavg.npy', similarity)
    logger.info("Successfully saving similarity matrix!")
    
    ## plot correlation
    figure = plt.figure(figsize=(5,5))
    axes = figure.add_subplot(111) 
    caxes = axes.matshow(similarity, interpolation ='nearest') 
    figure.colorbar(caxes) 
    plt.savefig(path_output + 'network/sample_neuron_similarity_3odors_tuning_trialavg.png',  dpi = 300, bbox_inches='tight')
    plt.show()


    # for odor in range(3):
    #     logger.info("------------- odor " + str(odor) + "-----------------")
    #     odor_stim = [i for i in range(len(stim)) if stim[i] == odor + 1]
        
    #     records_sample = np.load(path_output +'neuron_concat_records/sample_neuron_records_3odors_tuning_odor_' + str(odor) + '.npy',allow_pickle=True)
    #     logger.info("records_sample: {}, {}".format(type(records_sample), records_sample.shape))

    #     # 01 generate functional connectivity matrix
        
    #     ## calculate pearson correlation
    #     edges_all = []
    #     weight_all = []
    #     n_neurons = len(records_sample)
    #     similarity = np.zeros((n_neurons, n_neurons))
    #     for i in range(n_neurons - 1):
    #         for j in range(i + 1, n_neurons):   
    #             pearson = pearsonr(records_sample[i], records_sample[j])
    #             similarity[i,j] = similarity[j,i] = pearson[0] 
    #             edges_all.append((i,j,round(pearson[0],3)))
    #             weight_all.append(round(pearson[0],3))
    #     logger.info("edges_all: {}, weight_all:{}".format(len(edges_all), len(weight_all)))
    #     np.save(path_output + 'network/sample_neuron_similarity_odor' + str(odor) + '_tuning_3odors.npy', similarity)
    #     logger.info("Successfully saving similarity matrix!")
        
    #     ## plot correlation
    #     figure = plt.figure(figsize=(5,5))
    #     axes = figure.add_subplot(111) 
    #     caxes = axes.matshow(similarity, interpolation ='nearest') 
    #     figure.colorbar(caxes) 
    #     plt.savefig(path_output + 'network/sample_neuron_similarity_odor' + str(odor) + '_tuning_3odors.png',  dpi = 300, bbox_inches='tight')
    #     plt.show()
            
    #     # 02 generate functional connectivity network
        
    #     ## keep only correlations with high positive values 
    #     delete_seg = 0.6
    #     weight_all.sort()
    #     threshold = weight_all[int(delete_seg * len(weight_all))]

    #     ## generate network from correlation matrix
    #     weighted_edges = [] 
    #     source = []
    #     target = []
    #     weight = []
    #     nodes = [i for i in range(n_neurons)]
    #     for edge in edges_all:
    #         if edge[2] < threshold:
    #             continue
    #         weighted_edges.append(edge)
    #         source.append(int(edge[0]))
    #         target.append(int(edge[1]))
    #         weight.append(np.round(edge[2],3))
    #     logger.info("finish generating network ...")
    #     logger.info("# original edges:{}, # preserved edges: {}".format(len(weight_all), len(weighted_edges)))

    #     ## save network into csv files
    #     save_data_calc = {"source": source, "target": target, "weight": weight}
    #     df = pd.DataFrame(save_data_calc)
    #     df.to_csv(path_output + 'network/sample_neuron_similarity_odor' + str(odor) + '_network_tuning_3odors.csv', index=False)

logger.info("END")
    
