import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy.stats import pearsonr
import networkx as nx
from networkx.algorithms import community
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
log_file_name = 'brain-network-ach-for-ach-flies'
logging_name = 'brain-network'
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

    focus_regions = []
    ## Some brain regions of the specified flies are not used
    if data_index in ['20230428-fly1', '20230510-fly1']:
        focus_regions = focus_regions_1
    elif data_index == '20230507-fly1':
        focus_regions = focus_regions_2
    else:
        focus_regions = focus_regions_0

    ## path
    path = '/sam2p/0-FJQ/olfactory_representation/data/' + date_id + '-nsyb-G7f-rAch1h/' + fly_id + '/data/'
    path_atlas  = '/sam2p/0-FJQ/olfactory_representation/data/' + date_id + '-nsyb-G7f-rAch1h/' + fly_id + '/align_to_atlas/'
    path_output = '/sam2p/0-LLB/olfactory_representation/Ach-' + data_index + '-trans-stimu/'

    # 02 average recordings for each region
    region_records_avg = []
    for region_idx in focus_regions:
        logger.info("---- processing region {} ----".format(region_idx))

        ## load record data
        records = np.load(path_output +'neuron_concat_records/region_' + str(region_idx) + '_neuron_records.npy',allow_pickle=True)
        logger.info("G7f data: {} {}".format(len(records), len(records[0])))
        
        ## select records for the given time period
        select_data = records[:, 3 : 23]
        frames = 20 
        duration = 23
        start_timepoint = 3 + duration
        len_max = len(records[0])
        while start_timepoint < len_max:
            select_data = np.concatenate((select_data,records[:, start_timepoint : start_timepoint + frames]), axis = 1) 
            start_timepoint += duration
        logger.info("selected data: {} {}".format(len(select_data), len(select_data[0])))
        
        ## generate average records
        select_data_avg = np.mean(select_data, axis = 0)
        region_records_avg.append(select_data_avg)
        
    ## plot average records
    figure = plt.figure(figsize=(5,5)) 
    data = pd.DataFrame(region_records_avg)
    sns.heatmap(data)
    plt.savefig(path_output + 'network/whole_brain_all_regions_records.png', dpi = 300, bbox_inches='tight')
    plt.show()

    # 03 generate functional connectivity matrix

    ## calculate pearson correlation matrix
    data_list = []
    n_neurons = len(region_records_avg)
    similarity = np.zeros((n_neurons, n_neurons))
    for i in range(n_neurons - 1):
        for j in range(i + 1, n_neurons):   
            pearson = pearsonr(region_records_avg[i], region_records_avg[j])
            similarity[i][j] = similarity[j][i] = pearson[0]
    similarity = np.nan_to_num(similarity)
    np.save(path_output + 'network/whole_brain_all_regions_similarity.npy', similarity)
    logger.info("finish calculating corr ...")

    ## plot correlation matrix
    figure = plt.figure(figsize=(5,5)) 
    axes = figure.add_subplot(111) 
    caxes = axes.matshow(similarity, interpolation ='nearest') 
    figure.colorbar(caxes) 
    plt.savefig(path_output + 'network/whole_brain_all_regions_similarity.png', dpi = 300, bbox_inches='tight')
    plt.show()

    # 04 generate functional connectivity network

    ## keep only correlations with high positive values 
    delete_seg = 0
    weighted_edges_list = similarity.flatten()
    weighted_edges_list.sort()
    threshold = weighted_edges_list[int(delete_seg * len(weighted_edges_list))]

    ## generate network from correlation matrix
    nodes = [i for i in range(len(focus_regions_0)) if focus_regions_0[i] in focus_regions]
    weighted_edges = [] 
    source = []
    target = []
    weight = []
    for i in range(n_neurons - 1):
        for j in range(i + 1, n_neurons):
            x, y = nodes[i], nodes[j]
            v = similarity[i,j]
            if v < threshold:
                continue
            weighted_edges.append(np.array([x, y, np.round(v,4)]))
            source.append(x)
            target.append(y)
            weight.append(np.round(v,4))
    logger.info("finish generating network ...")
    logger.info("# original edges: {}, # preserved edges: {}".format(len(weighted_edges_list), len(weighted_edges)))

    ## save network into csv files
    save_data_calc = {"source": source, "target": target, "weight": weight}
    df = pd.DataFrame(save_data_calc)
    df.to_csv(path_output + 'network/regions_of_whole_brain_network_edges' + str(round(1 - delete_seg, 1)) + '.csv', index=False)

    # 05 calculate network statistics

    ## generate network
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(weighted_edges)

    ## network statistics
    n_nodes, n_edges = G.number_of_nodes(),G.number_of_edges()
    logger.info("# node: {}, # edges: {}".format(n_nodes, n_edges))
    degrees = nx.degree_histogram(G)
    # x = range(len(degrees))
    # y = [z / float(sum(degrees)) for z in degrees]
    # plt.loglog(x,y, linewidth = 2)
    # plt.xlabel("node degree", fontdict={'size': 16})
    # plt.ylabel("Proportion", fontdict={'size': 16})
    # plt.show()

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

    degree_centrality = [round(i,3) for i in nx.degree_centrality(G).values()]
    degree_centrality_avg = np.mean(degree_centrality)

    closeness_centrality = [round(i,3) for i in nx.closeness_centrality(G).values()]
    closeness_centrality_avg = np.mean(closeness_centrality)

    betweenness_centrality = [round(i,3) for i in nx.betweenness_centrality(G).values()]
    betweenness_centrality_avg = np.mean(betweenness_centrality)

    louvain_communities_comm = community.louvain_communities(G, weight='weight')  
    louvain_communities_mod = community.modularity(G, louvain_communities_comm)

    greedy_modularity_communities_comm = community.greedy_modularity_communities(G, weight='weight')  
    greedy_modularity_communities_mod = community.modularity(G, greedy_modularity_communities_comm)

    ## save network statistics into csv files 
    #writer = pd.ExcelWriter(path_output +'network_stat/regions_of_whole_brain_network_edges' + str(round(1 - delete_seg, 1)) + '_stat.xlsx')  
    data_dict = {}
    data_dict["stats"] = ["nodes", "edges", "degree_assortativity", "avg_clustering", "avg_shortest_path", "density", "diameter", 
                        "transitivity", "avg_degree","n_degrees_mean", "n_degrees_mean_avg", "n_avg_degree_centrality", 
                        "n_avg_closeness_centrality", "n_avg_betweenness_centrality","n_degrees", "n_degree_centrality", 
                        "n_closeness_centrality",  "n_betweenness_centrality", "louvain_modulairty", "greedy_modularity"]
    data_dict["values"] = [n_nodes, n_edges, round(degree_assortativity,3),round(avg_clustering,3),round(avg_shortest_path,3), round(density,3),
                        diameter, round(transitivity,3), round(avg_degree,3),
                        node_degrees_avg, round(node_degrees_avg/n_nodes,3), degree_centrality_avg,
                        closeness_centrality_avg, betweenness_centrality_avg, 
                        node_degrees, degree_centrality, closeness_centrality, betweenness_centrality,
                        louvain_communities_mod, greedy_modularity_communities_mod]
    df = pd.DataFrame(data_dict)
    file_name = path_output +'network_stat/regions_of_whole_brain_network_edges' + str(round(1 - delete_seg, 1)) + '_stat.xlsx'
    with pd.ExcelWriter(file_name, engine = 'openpyxl') as writer:
        df.to_excel(writer, sheet_name = "stat", index = False)        
    #writer.save()
    logger.info("finish saving stat ...")

logger.info('END')
