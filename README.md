This repo contains all 1-time codes required for SMD related processing
* `ServerMachineDataset` folder contains the original dataset from the authors.
* Use `create_smd_attack_dataset.py` and `create_smd_normal_dataset.py` to convert the 
data files from txt to csv appropriately.
* The resultant dataset is `smd`.
* `initial_eda_for_smd.py` contains the initial EDA code for SMD including obtaining all-zero counts for columns,
preparatory codes for box plots, and the necessary codes for pairplots
* `boxplots_varianceOfMean_pairplots.ipynb` is the Jupyter notebook for obtaining necessary visualization plots.

kmeans clustering code :
* `kmeans_clustering.py` contains the code to perform kmeans clustering on SMD dataset, and the Elbow method.
* `pair-wise-euclidean-dist.py` contains the code to obtain pairwise Euclidean distance between machines in cluster to 
form similarity chain

kruskal based clustering code :
* `similarity-measure-clustering.py` - Main code to obtain JS_distance, cluster based on JSD and derive similarity chains
* `kruskal_algo.py` - Kruskal-based clustering code
* `mst_using_kruskal.py` - MST to obtain similarity chains
