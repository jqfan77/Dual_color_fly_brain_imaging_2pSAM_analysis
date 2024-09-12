import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import pearsonr
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from itertools import chain
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
focus_regions_0 = [64,65,66,55,72,73,74,63,84,59,4,23,26,56,79,75,76,77,60,80,82,85,67]

# logger
log_file_name = 'temporal-network-ach'
logging_name = 'temporal-network'
today = datetime.datetime.now()
formatted_date = today.strftime('%Y-%m-%d')
logger = logger_config(log_path = 'log_' + log_file_name + "_" + formatted_date + '.log', logging_name = logging_name)
logger.info('Start the code ...')


''' 01 generate average correlation matrix '''

## path
path = '/as13000_sam2p/0-LLB/olfactory_representation/'
outpath = path + 'ACh_5HT_average_Stim/'

period_cnt = 4
period_trails = 180 / period_cnt
whole_start_timepoint = 3
for period_i in range(period_cnt):
    logger.info('-------------- period ' + str(period_i) + ' -------------')

    # initialization
    region_similarity_dict = {}
    n_regions = len(focus_regions_0)
    for i in range(n_regions - 1):
        for j in range(i + 1, n_regions):   
            region_similarity_dict[(focus_regions_0[i],focus_regions_0[j])] = []
    region_records_avg_dict = {}
    for i in range(n_regions):
        region_records_avg_dict[focus_regions_0[i]] = []

    '''
    process ACh flies
    '''
    
    data_index_list = ['20230417-fly2', '20230420-fly2', '20230420-fly3', '20230428-fly1', '20230507-fly1', '20230510-fly1',
                       '20230510-fly2', '20230511-fly2', '20230511-fly3', '20230515-fly1']
    focus_regions_0 = [64,65,66,55,72,73,74,63,84,59,4,23,26,56,79,75,76,77,60,80,82,85,67]
    focus_regions_1 = [64,65,66,55,72,73,74,63,84,59,4,23,26,56,79,75,76,77,60,85,67] # 80,82,
    focus_regions_2 = [64,65,66,72,73,74,63,84,59,4,23,26,56,79,75,76,77,60,85,67] # 55,80,82,

    for fly_idx in data_index_list:
        inputpath = path + 'Ach-' + fly_idx + '-trans-stimu/'
        logger.info("---- processing inputpath: " + inputpath)

        region_records_avg = []
        focus_regions = []
        if fly_idx == '20230428-fly1' or fly_idx == '20230510-fly1':
            focus_regions = focus_regions_1
        elif fly_idx == '20230507-fly1':
            focus_regions = focus_regions_2
        else:
            focus_regions = focus_regions_0

        for region_idx in focus_regions:  
            ## load record data
            records = np.load(inputpath +'neuron_concat_records/region_' + str(region_idx) +'_neuron_records.npy',allow_pickle=True)

            ## select records for the given time period
            cnt = 45 * period_i + 1 
            frames = 20
            duration = 23
            start_timepoint = whole_start_timepoint
            pre_sti = records[:, start_timepoint : start_timepoint + frames]
            start_timepoint += duration
            len_max = len(records[0])
            while cnt < 45 * (period_i + 1):
                cnt += 1
                pre_sti = np.concatenate((pre_sti,records[:, start_timepoint : start_timepoint + frames]), axis = 1) 
                start_timepoint += duration    

            ## generate average records
            pre_sti_avg = np.mean(pre_sti, axis = 0)

            region_records_avg.append(pre_sti_avg)
            region_records_avg_dict[region_idx].append(pre_sti_avg)
        
        ## calculate correlation
        n_regions_new = len(focus_regions)
        for i in range(n_regions_new - 1):
            for j in range(i + 1, n_regions_new):   
                pearson = pearsonr(region_records_avg[i], region_records_avg[j])
                pair = (focus_regions[i], focus_regions[j])
                region_similarity_dict[pair].append(pearson[0])
    whole_start_timepoint = start_timepoint


    '''
    calculate correlation matrix
    '''
    similarity = np.zeros((n_regions, n_regions))
    for i in range(n_regions - 1):
        for j in range(i + 1, n_regions):   
            similarity[i,j] = similarity[j,i] = np.mean(region_similarity_dict[(focus_regions_0[i],focus_regions_0[j])])
    np.save(outpath + 'whole_brain_all_regions_similarity_ACH_preiod' + str(period_i) + '.npy', similarity)
    logger.info("similarity:{}".format(similarity.shape))

    figure = plt.figure(figsize=(8,8)) 
    axes = figure.add_subplot(111) 
    caxes = axes.matshow(similarity, interpolation ='nearest') 
    figure.colorbar(caxes) 
    plt.savefig(outpath + 'whole_brain_all_regions_similarity_ACH_preiod' + str(period_i) + '.png')
    plt.show()
    logger.info("---- Finish saving similarity matrix!")


    '''
    calculate hierarchy clustering
    '''
    region_records_avg_list = []
    for i in range(n_regions):
        records_avg = np.mean(region_records_avg_dict[focus_regions_0[i]], axis = 0)
        region_records_avg_list.append(records_avg) 
    logger.info("region_records_avg_list:{}".format(np.array(region_records_avg_list).shape))
    np.save(outpath + 'whole_brain_all_regions_records_avg_ACH_preiod' + str(period_i) + '.npy', region_records_avg_list)

    ## clusters
    Z = linkage(region_records_avg_list, 'ward') 
    plt.figure(figsize=(50, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(Z, truncate_mode='lastp', p=12, leaf_rotation=90., leaf_font_size=12., show_contracted=True)
    plt.show()

    k = 2
    clusters = fcluster(Z, k, criterion='maxclust')
    logger.info("{},{},{}".format(type(clusters), len(clusters), clusters))
    list(enumerate(clusters))
    hierarchy_idx = []
    hierarchy_region = []
    for index_id in range(1,k+1):
        index = [i for i,j in enumerate(clusters) if j == index_id]
        logger.info("{},{},{}".format(index_id, len(index), index[:5]))
        hierarchy_idx.append(index)   
        hierarchy_region += list(np.array(focus_regions_0)[index])

    ## new region order
    hierarchy_region_name = []
    for i in hierarchy_region:
        hierarchy_region_name.append(region_names[i])
    hierarchy_idx_list = list(chain(*hierarchy_idx))

    ## calculate hierarchy correlation
    similarity = np.zeros((n_regions, n_regions))
    for i in range(n_regions - 1):
        for j in range(i + 1, n_regions):   
            pair = (hierarchy_region[i],hierarchy_region[j])
            if pair not in region_similarity_dict.keys():
                pair = (hierarchy_region[j],hierarchy_region[i])
            similarity[i,j] = similarity[j,i] = np.mean(region_similarity_dict[pair])
    logger.info("similarity:{}".format(similarity.shape))

    ## plot
    figure = plt.figure(figsize=(8,8)) 
    axes = figure.add_subplot(111) 
    caxes = axes.matshow(similarity, interpolation ='nearest') 
    figure.colorbar(caxes) 
    plt.xticks(rotation = 70)
    plt.xticks(range(len(hierarchy_region)), hierarchy_region_name)
    plt.yticks(rotation = 0)
    plt.yticks(range(len(hierarchy_region)), hierarchy_region_name)
    plt.savefig(outpath + 'whole_brain_all_regions_similarity_hierarchy_ACH_preiod' + str(period_i) + '.png')
    plt.show()

    hierarchy_idx_dict = {}
    for k in range(len(hierarchy_idx)):
        hierarchy_idx_dict[k] = hierarchy_idx[k]
    np.save(outpath + 'whole_brain_all_regions_similarity_hierarchy_ACH_idx_preiod' + str(period_i) + '.npy', hierarchy_idx_dict)
    np.save(outpath + 'whole_brain_all_regions_similarity_hierarchy_ACH_preiod' + str(period_i) + '.npy', similarity)
    logger.info("---- Finish saving hierarchy similarity matrix!")

logger.info("01 END")


''' 02 generate average network and communities '''

## parameters
delete_seg = 0
period_cnt = 4

for period_i in range(period_cnt):
    ## load data
    load_path = path + 'ACh_5HT_average_Stim/'
    similarity = np.load(load_path + 'whole_brain_all_regions_similarity_ACH_preiod' + str(period_i) + '.npy')
    logger.info("similarity:{}".format(similarity.shape))

    ## collect nodes and edges
    n_neurons = len(similarity)
    nodes = [i for i in range(n_neurons)]
    edges = {}
    for i in range(n_neurons - 1):
        for j in range(i + 1, n_neurons):   
            edges[(i,j)] = similarity[i,j]

    ## keep only correlations with high positive values 
    weighted_edges_list = []
    for v in edges.values():
        weighted_edges_list.append(v)
    weighted_edges_list.sort()
    threshold = weighted_edges_list[int(delete_seg * len(weighted_edges_list))]

    ## generate network from correlation matrix
    source, target, weight = [], [], []
    selected_edges = []
    for k,v in edges.items():
        if np.mean(v) < threshold:
            continue
        source.append(int(k[0]))
        target.append(int(k[1]))
        weight.append(np.round(v,4))
        selected_edges.append((int(k[0]), int(k[1]), np.round(v,4)))

    ## save network into csv files
    save_data_calc = {"source": source, "target": target, "weight": weight}
    df = pd.DataFrame(save_data_calc)
    df.to_csv(load_path + 'whole_brain_network_edges' + str(round(1 - delete_seg, 1)) + '_ACH_preiod' + str(period_i) + '.csv', index=False)

logger.info("02 END")


''' 03 calculate network statistics '''

## parameters
delete_seg = 0
nodes = [i for i in range(23)]

for period_i in range(4):
    calc_path = path + 'ACh_5HT_average_Stim/'
    similarity = np.load(calc_path + 'whole_brain_all_regions_similarity_ACH_preiod' + str(period_i) + '.npy')
    similarity_fla = similarity.flatten()

    ## edges
    similarity_fla.sort()
    threshold = similarity_fla[int(delete_seg * len(similarity_fla))]
    weighted_edges = [] 
    for i in range(len(similarity) - 1):
        for j in range(i + 1, len(similarity)):
            wei = np.round(similarity[i,j],4)
            if wei < threshold:
                continue
            weighted_edges.append(np.array([i,j,wei]))

    ## generate network
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(weighted_edges)
    n_nodes, n_edges = len(nodes), len(weighted_edges)

    ## network stats
    degree_assortativity = nx.degree_assortativity_coefficient(G)
    avg_clustering = nx.average_clustering(G)
    avg_shortest_path = 0
    diameter = 0
    try:
        avg_shortest_path = nx.average_shortest_path_length(G)
        diameter = nx.diameter(G)
    except:
        logger.info('G is not connected!')
    density = nx.density(G)
    transitivity = nx.transitivity(G)
    d = dict(nx.degree(G))
    avg_degree = sum(d.values())/G.number_of_nodes()
    node_degrees = [round(i[1],3) for i in nx.degree(G,weight="weight")]
    node_degrees_avg = np.mean(node_degrees)
    node_degrees.sort(reverse = True)

    degree_centrality = [round(i,3) for i in nx.degree_centrality(G).values()]
    degree_centrality_avg = np.mean(degree_centrality)
    degree_centrality.sort(reverse = True)
    
    ## save stats into file
    data_dict = {}
    data_dict["stats"] = ["nodes", "edges", "degree_assortativity", "avg_clustering", "avg_shortest_path", "density", "diameter", "transitivity", "avg_degree",
           "n_degrees_mean", "n_degrees_mean_avg", "n_avg_degree_centrality", "n_degrees", "n_degree_centrality"]
    data_dict["calc"] = [n_nodes, n_edges, round(degree_assortativity,3),round(avg_clustering,3),round(avg_shortest_path,3), round(density,3),
                        diameter, round(transitivity,3), round(avg_degree,3),
                        node_degrees_avg, round(node_degrees_avg/n_nodes,3), degree_centrality_avg, node_degrees, degree_centrality]
    df = pd.DataFrame(data_dict)
    file_name = calc_path +'whole_brain_network_edges' + str(round(1 - delete_seg, 1)) + '_stat_ACH_preiod' + str(period_i) + '.xlsx'
    with pd.ExcelWriter(file_name, engine = 'openpyxl') as writer:
        df.to_excel(writer, sheet_name = "stat", index = False)        
    #writer.save()
    logger.info("finish saving stat ...")
    
    
logger.info("03 END")


