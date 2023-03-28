# Obtain the pair-wise Euclidean distance between the means of 14 good machines
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

# Read from dataframe and create numpy array
column_value_dict = {}
df = pd.read_csv('EDA_results/5_selected_columns_for_14_machines.csv')

for i in range(len(df)):
    column_value_dict[df.iloc[i]['Machine']]=(np.array([df.iloc[i]['col6'], df.iloc[i]['col7'], df.iloc[i]['col23'], df.iloc[i]['col24']]))

print(column_value_dict)

# Initialisation A

# Consider 2 clusters, cluster1 and cluster2. Obtain the pairwise Euclidean distance between elements in that cluster.
cluster1=['machine-1-2', 'machine-1-5', 'machine-2-8']
cluster2=['machine-1-6', 'machine-2-5', 'machine-2-7', 'machine-3-1', 'machine-3-9']
cluster1_pairs = [(a, b) for idx, a in enumerate(cluster1) for b in cluster1[idx + 1:]]
cluster2_pairs = [(a, b) for idx, a in enumerate(cluster2) for b in cluster2[idx + 1:]]

print(cluster1_pairs)
print(cluster2_pairs)
print("\n")
for pair in cluster1_pairs:
    print(pair)
    print(euclidean_distances([column_value_dict[pair[0]]],[column_value_dict[pair[1]]]))

print("\n")

pair_dist_dict = {}
for pair in cluster2_pairs:
    print(pair)
    dist = euclidean_distances([column_value_dict[pair[0]]],[column_value_dict[pair[1]]])
    print(dist)
    pair_dist_dict[pair]=dist
print(pair_dist_dict)
print(dict(sorted(pair_dist_dict.items(), key=lambda item: item[1])))

# Initialisation B

cluster1_centre = [[0.72866857, 0.09642973, 0.05359851, 0.05545216]]
cluster2_centre = [[0.59613319, 0.26618271, 0.17094535, 0.5154326]]
print("\n")
for machine in cluster1:
    print(machine)
    print(euclidean_distances(cluster1_centre, [column_value_dict[machine]]))

print("\n")
centre_dist_dict = {}
for machine in cluster2:
    print(machine)
    centre_dist = euclidean_distances(cluster2_centre, [column_value_dict[machine]])
    print(centre_dist)
    centre_dist_dict[machine]=centre_dist
print(centre_dist_dict)
print(dict(sorted(centre_dist_dict.items(), key=lambda item: item[1])))
# [[0.913552330243368, 0.2622738759787772, 0.648472176586216, 0.0100638911829769], [0.6139201597450785, 0.2009237928167489, 0.0366851589853929, 0.0701442154975936], [0.5756051789917749, 0.0529418553891572, 0.1218392006749626, 0.0447194753427538], [0.7749818392434972, 0.3018336998058105, 0.1726113668524147, 0.3649833400878217], [0.920213976368332, 0.5926686691142375, 0.4430648625564407, 0.7622874824239682], [0.351452597306652, 0.2305047569655524, 0.2376919616683547, 0.4643341978216824], [0.3577744943028341, 0.2327475528359198, 0.237791477506751, 0.4743288367235007], [0.9964803826259756, 0.0354235306725169, 0.0022711735718505, 0.0514927858408574], [0.7600009483972128, 0.2795543611149839, 0.1045002707317074, 0.6352795141463388], [0.9905744513023218, 0.985152166758141, 0.4603412724279116, 0.2226372131126701], [0.9852602785281028, 0.024590792870559, 0.7419460364826351, 0.0125769848221117], [0.7364560530421704, 0.2862731798836745, 0.102131680110054, 0.638237112527434], [0.5324617867212692, 0.8546314935421165, 0.0933268191372612, 0.7710046470960262], [0.986080759435444, 0.96777439505139, 0.3528286897020496, 0.5389220983098281]]
# ['machine-1-1', 'machine-1-2', 'machine-1-5', 'machine-1-6', 'machine-1-7', 'machine-2-5', 'machine-2-7', 'machine-2-8', 'machine-3-1', 'machine-3-4', 'machine-3-6', 'machine-3-9', 'machine-3-10', 'machine-3-11']
