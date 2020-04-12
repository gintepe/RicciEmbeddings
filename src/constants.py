WEIGHT_ATTRIBUTE = "weight"
CAT = 'category'
LABELS_CORA = ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks', 'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory']
LABELS_FB = ['tvshow', 'government', 'politician', 'company']
FB_NET_PATH = "/home/ginte/dissertation/diss/data/fb_class/musae_facebook_edges.csv"
FB_TARGET_PATH = "/home/ginte/dissertation/diss/data/fb_class/musae_facebook_target.csv"


# Network Alignment related
ORIG_ATTRIBUTE = "original"
ARGS_EMBEDDING = "args"
SPECTRAL_EMBEDDING = "spectral"
RICCI_FLOW_METRIC = "flow"
HOP_COUNT = "hops"

# NMF related
RICCI_MATRIX = 'ricci'
HALF_RICCI_MATRIX = 'half'
ADJACENCY_MATRIX = 'adj'

RICCI_ALPHA = 0.5
TRANSFORMATION_ALPHA = 4

MATRIX_TYPES = [ADJACENCY_MATRIX, RICCI_MATRIX, HALF_RICCI_MATRIX]