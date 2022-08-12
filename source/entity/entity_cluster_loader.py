from entity.entity_cluster import EntityCluster
from utils.utils import parse_entity_cluster
from utils.config import *

entity_clusters = list()
with open(ENTITY_CLUSTER_INFO_FILE, "rt") as entity_cluster_file:
    for line in entity_cluster_file:
        entity_clusters.append(line.strip())


def get_entity_cluster(id):
    text = entity_clusters[int(id)]
    parts = text.split("\t")
    return EntityCluster(parts[0], None, parts[1], parse_entity_cluster(parts[2]), None)
