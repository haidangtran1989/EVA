import os

# Training
MAX_QUERY_LEN = 30
MAX_PASSAGE_LEN = 200
MS_MARCO_TRAINING_SIZE = 300000
TRAINING_EPOCHS = 2
LOCAL_BATCH_SIZE = 128
INITIAL_LEARNING_RATE = 3e-5
USE_KNRM = (os.environ["USE_KNRM"] == "yes")
if USE_KNRM:
    POS_NEG_GAP_LIMIT = -0.05
else:
    POS_NEG_GAP_LIMIT = -0.3
BASE_MODEL_LINK = "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
END_TOKEN_ID = 102

# Folders
EVA_MODEL_CHECKPOINT_DIR = "../models/eva"
if USE_KNRM:
    EVA_MODEL_CHECKPOINT_DIR += "-knrm"
    INDEX_FILE = "../index/eva_multi_knrm/eva_multi_knrm.index"
    ENTITY_CLUSTER_INFO_FILE = "../index/eva_multi_knrm/eva_multi_knrm.entity.clusters.tsv"
else:
    INDEX_FILE = "../index/eva_multi/eva_multi.index"
    ENTITY_CLUSTER_INFO_FILE = "../index/eva_multi/eva_multi.entity.clusters.tsv"
EVA_MODEL_CHECKPOINT = EVA_MODEL_CHECKPOINT_DIR + "/epoch-" + str(TRAINING_EPOCHS) + ".ckpt"
WIKIVEC_MODEL_FILE = "../data/wikipedia2vec-text-20210220.txt.gz"

# Index
NUMBER_CLUSTERS = 25000  # for multi representations, this is 4 * sqrt(N) with N = 41M
BERT_DIMENSION = 768
ENTITY_DIMENSION = 100
CLUSTER_LOOK_UP_NUMBER = 1000
KERNEL_DIMENSION_INPUT = 7
KERNEL_DIMENSION_OUTPUT = 1
TOTAL_DIMENSION = BERT_DIMENSION + ENTITY_DIMENSION
if USE_KNRM:
    TOTAL_DIMENSION += KERNEL_DIMENSION_OUTPUT

# Others
WIKI_PATTERN = "<en link=\"http://en.wikipedia.org/wiki/"
ENTITY_END_PATTERN = "\">"
TAG_END_PATTERN = "</en>"
SPLIT_LINE = "+++++"
ENTITY_PREFIX = "ENTITY/"
TOP_RANKED_RESULTS_FOR_DISPLAY = 10
TOP_RANKED_RESULTS_FOR_EVALUATION = 1000
TOP_RANKED_CLUSTER_RESULTS = 3800
MULTI_TITLE = "EVA Multi"
if USE_KNRM:
    MULTI_TITLE += "-KNRM"
