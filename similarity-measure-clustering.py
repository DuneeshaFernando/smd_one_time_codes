import pandas as pd
from sklearn import preprocessing
from itertools import combinations
from scipy.spatial import distance
from kruskal_algo import *
from mst_using_kruskal import *
import numpy as np

min_max_scaler = preprocessing.MinMaxScaler()

# Obtain the 4 distributions for the 14 machines w/o concept drift
normal_dataset_path = 'smd/train/'
machine_files = ['machine-1-1.csv', 'machine-1-2.csv', 'machine-1-5.csv', 'machine-1-6.csv', 'machine-1-7.csv', 'machine-2-5.csv', 'machine-2-7.csv', 'machine-2-8.csv', 'machine-3-1.csv', 'machine-3-4.csv', 'machine-3-6.csv', 'machine-3-9.csv', 'machine-3-10.csv', 'machine-3-11.csv']
# Following machine_files lists are used when obtaining similarity chains for individual clusters
# machine_files = ['machine-1-5.csv', 'machine-2-8.csv', 'machine-1-2.csv']
# machine_files = ['machine-3-1.csv', 'machine-3-9.csv', 'machine-1-6.csv', 'machine-3-10.csv', 'machine-1-7.csv']
selected_cols = ['col6','col7','col23','col24']
normal_df_list = [pd.read_csv(normal_dataset_path + normal_data_file, usecols=selected_cols) for normal_data_file in machine_files]

# Create dictionary by numbering machines in machine_files
machine_diction = {}
machine_diction_index = 0
for machine in machine_files:
    machine_diction[machine_diction_index]=machine.split(".csv")[0]

# Normalise the dataset
# First concat all dataframes to fit
normal_df = pd.concat(normal_df_list)
normal_df = normal_df.astype(float)

min_max_scaler.fit(normal_df)

pairwise_jsd_list = []
# Obtain pair-wise combinations of machines
for comb in combinations([i for i in range(len(machine_files))], 2):
    machine_1_df = pd.read_csv(normal_dataset_path + machine_files[comb[0]], usecols=selected_cols)
    machine_1_df_values = machine_1_df.values
    machine_1_df_values_scaled = min_max_scaler.transform(machine_1_df_values)

    machine_2_df = pd.read_csv(normal_dataset_path + machine_files[comb[1]], usecols=selected_cols)
    machine_2_df_values = machine_2_df.values
    machine_2_df_values_scaled = min_max_scaler.transform(machine_2_df_values)

    machine_1_col6 = pd.cut(machine_1_df_values_scaled[:, 0], [round(i * 0.1, 2) for i in range(11)]).value_counts()
    machine_1_col7 = pd.cut(machine_1_df_values_scaled[:, 1], [round(i * 0.1, 2) for i in range(11)]).value_counts()
    machine_1_col23 = pd.cut(machine_1_df_values_scaled[:, 2], [round(i * 0.1, 2) for i in range(11)]).value_counts()
    machine_1_col24 = pd.cut(machine_1_df_values_scaled[:, 3], [round(i * 0.1, 2) for i in range(11)]).value_counts()

    machine_1_col6[:] = machine_1_col6[:] / machine_1_col6.sum()
    machine_1_col7[:] = machine_1_col7[:] / machine_1_col7.sum()
    machine_1_col23[:] = machine_1_col23[:] / machine_1_col23.sum()
    machine_1_col24[:] = machine_1_col24[:] / machine_1_col24.sum()

    machine_2_col6 = pd.cut(machine_2_df_values_scaled[:, 0], [round(i * 0.1, 2) for i in range(11)]).value_counts()
    machine_2_col7 = pd.cut(machine_2_df_values_scaled[:, 1], [round(i * 0.1, 2) for i in range(11)]).value_counts()
    machine_2_col23 = pd.cut(machine_2_df_values_scaled[:, 2],
                             [round(i * 0.1, 2) for i in range(11)]).value_counts()
    machine_2_col24 = pd.cut(machine_2_df_values_scaled[:, 3],
                             [round(i * 0.1, 2) for i in range(11)]).value_counts()

    machine_2_col6[:] = machine_2_col6[:] / machine_2_col6.sum()
    machine_2_col7[:] = machine_2_col7[:] / machine_2_col7.sum()
    machine_2_col23[:] = machine_2_col23[:] / machine_2_col23.sum()
    machine_2_col24[:] = machine_2_col24[:] / machine_2_col24.sum()

    data_machine_1 = pd.concat([machine_1_col6, machine_1_col7, machine_1_col23, machine_1_col24], axis=1)
    data_machine_2 = pd.concat([machine_2_col6, machine_2_col7, machine_2_col23, machine_2_col24], axis=1)

    np_machine_1 = data_machine_1.to_numpy()
    np_machine_2 = data_machine_2.to_numpy()

    pairwise_jsd_list.append([comb[0], comb[1], distance.jensenshannon(np_machine_1, np_machine_2, axis=0)])

# print(pairwise_jsd_list)

# This 1st code is to obtain Kruskal based clusters

g = Graph(len(machine_files))
for entry in pairwise_jsd_list:
    print(entry)
    d1 = entry[2][0] ** 2
    d2 = entry[2][1] ** 2
    d3 = entry[2][2] ** 2
    d4 = entry[2][3] ** 2
    d = np.sqrt(d1 + d2 + d3 + d4)
    g.addEdge(entry[0], entry[1], d)

cluster_dict, mach_cluster_dict = g.KruskalMSTClusters(K=5)

for i in range(len(cluster_dict)):
    new_list = []
    for item in cluster_dict[i]:
        new_list.append(machine_files[item].strip('.csv'))
    cluster_dict[i]=new_list
print(cluster_dict)

for i in range(len(mach_cluster_dict)):
    mach_cluster_dict[machine_files[i]] = mach_cluster_dict.pop(str(i))
print(mach_cluster_dict)

# # This 2nd code is to obtain the similarity chains within individual clusters. Select correct machine_files list from above
#
# mstg = MSTGraph(len(machine_files))
# for entry in pairwise_jsd_list:
#     print(entry)
#     d1 = entry[2][0] ** 2
#     d2 = entry[2][1] ** 2
#     d3 = entry[2][2] ** 2
#     d4 = entry[2][3] ** 2
#     d = np.sqrt(d1 + d2 + d3 + d4)
#     mstg.addEdge(entry[0], entry[1], d)
#
# mst_final_list = mstg.KruskalMST()
# print(mst_final_list)
# for item in mst_final_list:
#     item[0]=machine_files[item[0]]
#     item[1] = machine_files[item[1]]
# print(mst_final_list)