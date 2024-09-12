import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import networkx as nx
from scipy.cluster.hierarchy import linkage, fcluster
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

data_index_list = ['20230429-r5HT1.0-fly1', '20230506-r5HT1.0-fly1', '20230513-r5HT1.0-fly1', 
                   '20230513-r5HT1.0-fly2', '20230516-r5HT1.0-fly2', '20230516-r5HT1.0-fly4',
                   '20230517-r5HT1.0-fly1', '20230601-r5HT1.0-fly1', '20230601-r5HT1.0-fly3', '20230603-r5HT1.0-fly1']
focus_regions_0 = [64,65,66,55,72,73,74,63,84,59,4,23,26,56,79,75,76,77,60,80,82,85,67] # all brain regions
focus_regions_1 = [64,65,66,55,72,73,74,63,84,59,4,23,26,56,79,75,76,77,60,80,85,67] # brain regions except 82
focus_regions_2 = [64,65,66,55,72,73,74,63,84,59,4,23,26,56,79,75,76,77,60,85,67] # brain regions except 80, 82
focus_regions_3 = [64,65,66,72,73,74,63,84,59,4,23,26,56,79,75,76,77,60,80,82,85,67] # brain regions except 55

# logger
log_file_name = 'region-network-g7f-for-5ht-flies'
logging_name = 'region-network'
today = datetime.datetime.now()
formatted_date = today.strftime('%Y-%m-%d')
logger = logger_config(log_path = 'log_' + log_file_name + "_" + formatted_date + '.log', logging_name = logging_name)
logger.info('Start the code ...')


for data_index in data_index_list: 
    logger.info('============ data_index ' + data_index + ' ===========')

    # 01 settiings

    ## parameters
    date_id = data_index.split('-')[0]
    fly_id = data_index.split('-')[2]
    n_trails = 180

    focus_regions = []
    ## Some brain regions of the specified flies are not used
    if data_index in ["20230429-r5HT1.0-fly1", "20230516-r5HT1.0-fly4"]:
        focus_regions = focus_regions_1
    elif data_index in ["20230601-r5HT1.0-fly3", "20230517-r5HT1.0-fly1"]:
        focus_regions = focus_regions_2
    elif data_index == "20230601-r5HT1.0-fly1":
        focus_regions = focus_regions_3
    else:
        focus_regions = focus_regions_0

    ## path
    path = '/as13000_sam2p/0-FJQ/olfactory_representation/data/' + date_id + '-nsyb-G7f-r5HT1.0/' + fly_id + '/data/'
    path_atlas  = '/as13000_sam2p/0-FJQ/olfactory_representation/data/' + date_id + '-nsyb-G7f-r5HT1.0/' + fly_id + '/align_to_atlas/'
    path_output = '/as13000_sam2p/0-LLB/olfactory_representation/' + date_id + '-r5HT1.0-' + fly_id + '-stimu/'

    ## generate network for each region
    for region_idx in [66,55,64,65,73,63,84,59,23,26,85,79,77]:   
        logger.info("---- processing region " + str(region_idx) + " ----")
        
        if region_idx not in focus_regions:
            continue
        
        ## load G7f data
        records = np.load(path_output + 'neuron_concat_records/region_' + str(region_idx) + '_neuron_records.npy',allow_pickle=True)
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
        np.save(path_output + 'network/region_' + str(region_idx) + '_all_neurons_records.npy', select_data)
        
        ## plot selected data
        figure = plt.figure(figsize=(5,3)) 
        data = pd.DataFrame(select_data)
        sns.heatmap(data)
        plt.savefig(path_output + 'network/region_' + str(region_idx) + '_all_neurons_records.png', dpi = 300, bbox_inches='tight')
        plt.show()
        
        # 02 generate functional connectivity matrix
        
        ## calculate pearson correlation
        region_similarity_dict = {}
        n_neurons = len(select_data)
        for i in range(n_neurons - 1):
            for j in range(i + 1, n_neurons):   
                pearson = pearsonr(select_data[i], select_data[j])
                region_similarity_dict[(i,j)] = pearson[0]         
        
        ## calculate hierarchy clustering
        Z = linkage(select_data, 'ward') 
        k = 4
        clusters = fcluster(Z, k, criterion='maxclust')
        list(enumerate(clusters))
        hierarchy_idx = []
        for index_id in range(1,k+1):
            index = [i for i,j in enumerate(clusters) if j == index_id]
            hierarchy_idx.append(index)   
        hierarchy_idx_list = list(chain(*hierarchy_idx))
        
        ## calculate hierarchy correlation
        similarity = np.zeros((n_neurons, n_neurons))
        for i in range(n_neurons - 1):
            for j in range(i + 1, n_neurons):   
                pair = (hierarchy_idx_list[i],hierarchy_idx_list[j])
                if pair not in region_similarity_dict.keys():
                    pair = (hierarchy_idx_list[j],hierarchy_idx_list[i])
                similarity[i,j] = similarity[j,i] = np.mean(region_similarity_dict[pair])
                
        ## plot correlation
        figure = plt.figure(figsize=(5,5))
        axes = figure.add_subplot(111) 
        caxes = axes.matshow(similarity, interpolation ='nearest') 
        figure.colorbar(caxes) 
        plt.savefig(path_output + 'network/region_' + str(region_idx) + '_all_neurons_similarity_hierarchy.png',  dpi = 300, bbox_inches='tight')
        plt.show()
        
        ## save hierarchy correlation
        hierarchy_idx = np.asarray(hierarchy_idx, dtype = object)
        np.save(path_output + 'network/region_' + str(region_idx) + '_all_neurons_similarity_hierarchy_idpart.npy', hierarchy_idx)
        np.save(path_output + 'network/region_' + str(region_idx) + '_all_neurons_similarity_hierarchy_ids.npy', hierarchy_idx_list)
        np.save(path_output + 'network/region_' + str(region_idx) + '_all_neurons_similarity_hierarchy.npy', similarity)
        logger.info("Successfully saving similarity matrix!")
            
        # 03 generate functional connectivity network
        
        ## keep only correlations with high positive values 
        delete_seg = 0.6
        weighted_edges_list = []
        for k,v in region_similarity_dict.items():
            weighted_edges_list.append(v)
        weighted_edges_list.sort()
        threshold = weighted_edges_list[int(delete_seg * len(weighted_edges_list))]

        ## generate network from correlation matrix
        weighted_edges = [] 
        source = []
        target = []
        weight = []
        nodes = [i for i in range(n_neurons)]
        for k,v in region_similarity_dict.items():
            if v < threshold:
                continue
            weighted_edges.append(np.array([int(k[0]), int(k[1]), np.round(v,4)]))
            source.append(int(k[0]))
            target.append(int(k[1]))
            weight.append(np.round(v,4))
        logger.info("finish generating network ...")
        logger.info("# original edges: {}, # preserved edges: {}".format(len(weighted_edges_list), len(weighted_edges)))

        ## save network into csv files
        save_data_calc = {"source": source, "target": target, "weight": weight}
        df = pd.DataFrame(save_data_calc)
        df.to_csv(path_output + 'network/region_' + str(region_idx) + '_network.csv', index=False)

        # 04 calculate network statistics
        
        ## generate network
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_weighted_edges_from(weighted_edges)

        ## network stats
        n_nodes, n_edges = G.number_of_nodes(),G.number_of_edges()
        logger.info("# node: {}, # edges: {}".format(n_nodes, n_edges))
        
        degree_assortativity = nx.degree_assortativity_coefficient(G)
        avg_clustering = nx.average_clustering(G)
        avg_shortest_path = 0
        try:
            avg_shortest_path = nx.average_shortest_path_length(G)
        except:
            logger.info('G is not connected!')
        d = dict(nx.degree(G))
        avg_degree = sum(d.values())/G.number_of_nodes()
        
        node_degrees = [round(i[1],3) for i in nx.degree(G,weight="weight")]
        node_degrees_avg = np.mean(node_degrees)
        node_degrees.sort(reverse = True)

        degree_centrality = [round(i,3) for i in nx.degree_centrality(G).values()]
        degree_centrality_avg = np.mean(degree_centrality)
        degree_centrality.sort(reverse = True)

        closeness_centrality = [round(i,3) for i in nx.closeness_centrality(G).values()]
        closeness_centrality_avg = np.mean(closeness_centrality)
        closeness_centrality.sort(reverse = True)
        
        ## save network statistics into csv files
        # writer = pd.ExcelWriter(path_output + 'network_stat/region_' + str(region_idx) + '_network_stat.xlsx')
        data_dict = {}
        data_dict["stats"] = ["nodes", "edges", "degree_assortativity", "avg_clustering", "avg_shortest_path", "avg_degree",
                            "n_degrees_mean", "n_avg_degree_centrality", "n_avg_closeness_centrality", "n_degrees", 
                            "n_degree_centrality", "n_closeness_centrality"]
        data_dict["values"] = [n_nodes, n_edges, round(degree_assortativity,3), round(avg_clustering,3), round(avg_shortest_path,3),
                    round(avg_degree,3), node_degrees_avg, degree_centrality_avg, closeness_centrality_avg, node_degrees, 
                    degree_centrality, closeness_centrality]

        df = pd.DataFrame(data_dict)
        file_name = path_output + 'network_stat/region_' + str(region_idx) + '_network_stat.xlsx'
        with pd.ExcelWriter(file_name, engine = 'openpyxl') as writer:
            df.to_excel(writer, sheet_name = "stat", index = False)        
        #writer.save()
        logger.info("finish saving stat ...")
        
logger.info("END")
