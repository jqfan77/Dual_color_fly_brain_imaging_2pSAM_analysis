import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
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
focus_regions_0 = [64,65,66,55,72,73,74,63,84,59,4,23,26,56,79,75,76,77,60,80,82,85,67]

# logger
log_file_name = 'temporal-network-g7f-single'
logging_name = 'temporal-network'
today = datetime.datetime.now()
formatted_date = today.strftime('%Y-%m-%d')
logger = logger_config(log_path = 'log_' + log_file_name + "_" + formatted_date + '.log', logging_name = logging_name)
logger.info('Start the code ...')

## path
path = '/sam2p/0-LLB/olfactory_representation/'


''' ach flies '''
data_index_list = ['20230417-fly2', '20230420-fly2', '20230420-fly3', '20230428-fly1', '20230507-fly1', '20230510-fly1',
                       '20230510-fly2', '20230511-fly2', '20230511-fly3', '20230515-fly1']
focus_regions_0 = [64,65,66,55,72,73,74,63,84,59,4,23,26,56,79,75,76,77,60,80,82,85,67]
focus_regions_1 = [64,65,66,55,72,73,74,63,84,59,4,23,26,56,79,75,76,77,60,85,67] # 80,82,
focus_regions_2 = [64,65,66,72,73,74,63,84,59,4,23,26,56,79,75,76,77,60,85,67] # 55,80,82,

for data_index in data_index_list:
    path_output = path + 'Ach-' + data_index + '-stimu/'
    logger.info("---- processing path_output: " + path_output)

    focus_regions = []
    ## Attention: Some brain regions of specified flies are not used.
    if data_index in ['20230428-fly1', '20230510-fly1']:
        focus_regions = focus_regions_1
    elif data_index == '20230507-fly1':
        focus_regions = focus_regions_2
    else:
        focus_regions = focus_regions_0
        
    period_cnt = 4
    period_trails = 180 / period_cnt
    whole_start_timepoint = 3
    for period_i in range(period_cnt):
        logger.info("---- processing period " + str(period_i) + " ----")

        region_records_avg = []
        for region_idx in focus_regions:  
            # load record data
            records = np.load(path_output +'neuron_concat_records/region_' + str(region_idx) + '_neuron_records.npy',allow_pickle=True)

            ## select records for the given time period
            cnt = period_trails * period_i + 1 
            frames = 20
            duration = 23
            start_timepoint = whole_start_timepoint
            selected_data = records[:, start_timepoint : start_timepoint + frames]
            start_timepoint += duration
            len_max = len(records[0])
            while cnt < period_trails * (period_i + 1):
                cnt += 1
                selected_data = np.concatenate((selected_data, records[:, start_timepoint : start_timepoint + frames]), axis = 1) 
                start_timepoint += duration   

            # generate average records
            selected_data_avg = np.mean(selected_data, axis = 0)
            region_records_avg.append(selected_data_avg)
            
        '''
        generate functional connectivity matrix
        '''

        ## calculate pearson correlation matrix
        region_similarity_dict = {}
        n_regions = len(focus_regions)
        similarity = np.zeros((n_regions, n_regions))
        for i in range(n_regions - 1):
            for j in range(i + 1, n_regions):   
                pearson = pearsonr(region_records_avg[i], region_records_avg[j])
                similarity[i,j] = similarity[j,i] = pearson[0]
                pair = (focus_regions[i],focus_regions[j])
                region_similarity_dict[pair] = pearson[0]
        np.save(path_output + 'network/whole_brain_all_regions_similarity_preiod' + str(period_i) + '.npy', similarity)

        ## plot correlation matrix
        figure = plt.figure(figsize=(5,5)) 
        axes = figure.add_subplot(111) 
        caxes = axes.matshow(similarity, interpolation ='nearest') 
        figure.colorbar(caxes) 
        plt.savefig(path_output + 'network/whole_brain_all_regions_similarity_preiod' + str(period_i) + '.png')
        plt.show()

        '''
        generate functional connectivity network
        '''

        ## keep only correlations with high positive values 
        delete_seg = 0
        similarity_fla = similarity.flatten()
        similarity_fla.sort()
        threshold = similarity_fla[int(delete_seg * len(similarity_fla))]

        ## generate network from correlation matrix
        weighted_edges = [] 
        source = []
        target = []
        weight = []
        for i in range(len(similarity) - 1):
            for j in range(i + 1, len(similarity)):
                wei = np.round(similarity[i,j],4)
                if wei < threshold:
                    continue
                weighted_edges.append(np.array([i,j,wei]))
                source.append(i)
                target.append(j)
                weight.append(wei)

        ## save network into csv files
        save_data_calc = {"source": source, "target": target, "weight": weight}
        df = pd.DataFrame(save_data_calc)
        df.to_csv(path_output + 'network/whole_brain_network_edges' + str(round(1 - delete_seg, 1)) + '_preiod' + str(period_i) + '.csv', index=False)

        '''
        calculate network statistics
        '''

        ## generate network  
        G = nx.Graph()
        G.add_weighted_edges_from(weighted_edges)

        ## network statistics
        n_nodes, n_edges = G.number_of_nodes(),G.number_of_edges()
        logger.info("# node: {}, # edges:{}".format(n_nodes, n_edges))
        degrees = nx.degree_histogram(G)
        x = range(len(degrees))
        y = [z / float(sum(degrees)) for z in degrees]
        plt.loglog(x,y, linewidth = 2)
        plt.xlabel("node degree", fontdict={'size': 16})
        plt.ylabel("Proportion", fontdict={'size': 16})
        plt.show()

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

        ## save network statistics into csv files
        data_dict = {}
        data_dict["stats"] = ["nodes", "edges", "degree_assortativity", "avg_clustering", "avg_shortest_path", "density", "diameter", "transitivity", "avg_degree",
            "n_degrees_mean", "n_degrees_mean_avg", "n_avg_degree_centrality", "n_degrees", "n_degree_centrality"]
        data_dict["calc"] = [n_nodes, n_edges, round(degree_assortativity,3),round(avg_clustering,3),round(avg_shortest_path,3), round(density,3),
                            diameter, round(transitivity,3), round(avg_degree,3),
                            node_degrees_avg, round(node_degrees_avg/n_nodes,3), degree_centrality_avg, node_degrees, degree_centrality]
        df = pd.DataFrame(data_dict)
        file_name = path_output +'network_stat/whole_brain_network_edges' + str(round(1 - delete_seg, 1)) + '_stat_preiod' + str(period_i) + '.xlsx'
        with pd.ExcelWriter(file_name, engine = 'openpyxl') as writer:
            df.to_excel(writer, sheet_name = "stat", index = False)        
        #writer.save()
        logger.info("finish saving stat ...")

        whole_start_timepoint = start_timepoint


''' 5ht flies '''
data_index_list = ['20230429-r5HT1.0-fly1', '20230506-r5HT1.0-fly1', '20230513-r5HT1.0-fly1', 
                    '20230513-r5HT1.0-fly2', '20230516-r5HT1.0-fly2', '20230516-r5HT1.0-fly4',
                    '20230517-r5HT1.0-fly1', '20230601-r5HT1.0-fly1', '20230601-r5HT1.0-fly3', '20230603-r5HT1.0-fly1']
focus_regions_0 = [64,65,66,55,72,73,74,63,84,59,4,23,26,56,79,75,76,77,60,80,82,85,67]
focus_regions_1 = [64,65,66,55,72,73,74,63,84,59,4,23,26,56,79,75,76,77,60,80,85,67] # 82
focus_regions_2 = [64,65,66,55,72,73,74,63,84,59,4,23,26,56,79,75,76,77,60,85,67] # 80,82
focus_regions_3 = [64,65,66,72,73,74,63,84,59,4,23,26,56,79,75,76,77,60,80,82,85,67] # 55

for data_index in data_index_list:
    path_output = path + data_index + '-stimu/'
    logger.info("---- processing path_output: " + path_output)

    focus_regions = []
    if data_index in ["20230429-r5HT1.0-fly1", "20230516-r5HT1.0-fly4"]:
        focus_regions = focus_regions_1
    elif data_index in ["20230601-r5HT1.0-fly3", "20230517-r5HT1.0-fly1"]:
        focus_regions = focus_regions_2
    elif data_index == "20230601-r5HT1.0-fly1":
        focus_regions = focus_regions_3
    else:
        focus_regions = focus_regions_0
        
    period_cnt = 4
    period_trails = 180 / period_cnt
    whole_start_timepoint = 3
    for period_i in range(period_cnt):
        logger.info("---- processing period " + str(period_i) + " ----")

        region_records_avg = []
        for region_idx in focus_regions:  
            # load record data
            records = np.load(path_output +'neuron_concat_records/region_' + str(region_idx) + '_neuron_records.npy',allow_pickle=True)

            ## select records for the given time period
            cnt = period_trails * period_i + 1 
            frames = 20
            duration = 23
            start_timepoint = whole_start_timepoint
            selected_data = records[:, start_timepoint : start_timepoint + frames]
            start_timepoint += duration
            len_max = len(records[0])
            while cnt < period_trails * (period_i + 1):
                cnt += 1
                selected_data = np.concatenate((selected_data, records[:, start_timepoint : start_timepoint + frames]), axis = 1) 
                start_timepoint += duration   

            # generate average records
            selected_data_avg = np.mean(selected_data, axis = 0)
            region_records_avg.append(selected_data_avg)
            
        '''
        generate functional connectivity matrix
        '''

        ## calculate pearson correlation matrix
        region_similarity_dict = {}
        n_regions = len(focus_regions)
        similarity = np.zeros((n_regions, n_regions))
        for i in range(n_regions - 1):
            for j in range(i + 1, n_regions):   
                pearson = pearsonr(region_records_avg[i], region_records_avg[j])
                similarity[i,j] = similarity[j,i] = pearson[0]
                pair = (focus_regions[i],focus_regions[j])
                region_similarity_dict[pair] = pearson[0]
        np.save(path_output + 'network/whole_brain_all_regions_similarity_preiod' + str(period_i) + '.npy', similarity)

        ## plot correlation matrix
        figure = plt.figure(figsize=(5,5)) 
        axes = figure.add_subplot(111) 
        caxes = axes.matshow(similarity, interpolation ='nearest') 
        figure.colorbar(caxes) 
        plt.savefig(path_output + 'network/whole_brain_all_regions_similarity_preiod' + str(period_i) + '.png')
        plt.show()

        '''
        generate functional connectivity network
        '''

        ## keep only correlations with high positive values 
        delete_seg = 0
        similarity_fla = similarity.flatten()
        similarity_fla.sort()
        threshold = similarity_fla[int(delete_seg * len(similarity_fla))]

        ## generate network from correlation matrix
        weighted_edges = [] 
        source = []
        target = []
        weight = []
        for i in range(len(similarity) - 1):
            for j in range(i + 1, len(similarity)):
                wei = np.round(similarity[i,j],4)
                if wei < threshold:
                    continue
                weighted_edges.append(np.array([i,j,wei]))
                source.append(i)
                target.append(j)
                weight.append(wei)

        ## save network into csv files
        save_data_calc = {"source": source, "target": target, "weight": weight}
        df = pd.DataFrame(save_data_calc)
        df.to_csv(path_output + 'network/whole_brain_network_edges' + str(round(1 - delete_seg, 1)) + '_preiod' + str(period_i) + '.csv', index=False)

        '''
        calculate network statistics
        '''

        ## generate network  
        G = nx.Graph()
        G.add_weighted_edges_from(weighted_edges)

        ## network statistics
        n_nodes, n_edges = G.number_of_nodes(),G.number_of_edges()
        logger.info("# node: {}, # edges:{}".format(n_nodes, n_edges))
        degrees = nx.degree_histogram(G)
        x = range(len(degrees))
        y = [z / float(sum(degrees)) for z in degrees]
        plt.loglog(x,y, linewidth = 2)
        plt.xlabel("node degree", fontdict={'size': 16})
        plt.ylabel("Proportion", fontdict={'size': 16})
        plt.show()

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

        ## save network statistics into csv files
        data_dict = {}
        data_dict["stats"] = ["nodes", "edges", "degree_assortativity", "avg_clustering", "avg_shortest_path", "density", "diameter", "transitivity", "avg_degree",
            "n_degrees_mean", "n_degrees_mean_avg", "n_avg_degree_centrality", "n_degrees", "n_degree_centrality"]
        data_dict["calc"] = [n_nodes, n_edges, round(degree_assortativity,3),round(avg_clustering,3),round(avg_shortest_path,3), round(density,3),
                            diameter, round(transitivity,3), round(avg_degree,3),
                            node_degrees_avg, round(node_degrees_avg/n_nodes,3), degree_centrality_avg, node_degrees, degree_centrality]
        df = pd.DataFrame(data_dict)
        file_name = path_output +'network_stat/whole_brain_network_edges' + str(round(1 - delete_seg, 1)) + '_stat_preiod' + str(period_i) + '.xlsx'
        with pd.ExcelWriter(file_name, engine = 'openpyxl') as writer:
            df.to_excel(writer, sheet_name = "stat", index = False)        
        #writer.save()
        logger.info("finish saving stat ...")

        whole_start_timepoint = start_timepoint

logger.info("END")