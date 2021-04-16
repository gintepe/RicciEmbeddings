# Ricci Curvature in Network Emedding and Clustering
## BSc Computer Science and Mathematics Year 4 Project

Final [report](https://project-archive.inf.ed.ac.uk/ug4/20201839/ug4_proj.pdf) published by the University of Edinburgh as part of the outstanding projects archive.

### Source code and results

The code is based on the source code for GEMSEC, (found at [1](https://github.com/benedekrozemberczki/GEMSEC)). The files from here have been seperated out into the *src/gemsec* forlder. The other files in the *src* folder represent the work which was added over the course of the project.

The `ricci_flow_explorations.py` and `ricci_flow_explorations_testing.py` files correspond to the experiments described in chapter 4, providing functions implementing spectral and MDS embeddings and the evaluation of post-embedding clustering.

In `classification.py` we have implemented logistic regression to evaluate classification results on networks, as well as some graph re-weighting strategies, training and test set selections, etc.

The `ricci_matrix.py` file allows to obtain clustering results via NMF (for different types of matrices obtained from the graphs), by either interpreting it as an embedding or using the resulting matrices as cluster centers and indicators.

In `network_alignment.py` we implement a network alignment framework, which allows to evaluate the performance of various metrics on aligning a graph to a deformed version of itself (to ensure a ground truth alignment is known). (results not included in the report due to time constraints)

The rest hold primarily various utility, visualization and testing functions.

We also include the raw results obtained (in the *res* folder) as well as datasets used (due to the reasonably small sizes). These include the datasets available together with GEMSEC, the CORA dataset [2](http://networkrepository.com/cora.php) and the MUSAE facebook dataset [3](https://snap.stanford.edu/data/facebook-large-page-page-network.html) (results on this dataset not included in the report)

Requires NetworkX, Numpy, Scipy, pandas, tqdm, Texttable, GraphRicciCurvature, tensorflow packages
