import numpy as np
import gzip
from utils.utils import normalize
from utils.config import *

entry_to_vector = dict()


with gzip.open(WIKIVEC_MODEL_FILE, "rt") as model_file:
    for line in model_file:
        parts = line.strip().split("\t")
        key = parts[0].replace(" ", "_")
        vector = np.fromstring(parts[1], dtype=np.float32, sep=" ")
        entry_to_vector[key] = vector


def get_entity_vector(entity):
    return get_vector(ENTITY_PREFIX + entity)


def get_vector(entry):
    entry_vector = np.zeros(ENTITY_DIMENSION)
    if entry in entry_to_vector:
        entry_vector = np.copy(entry_to_vector[entry])
    return entry_vector


def get_norm_vector(entry):
    return normalize(get_vector(entry))


def get_entity_similarity(entity1, entity2):
    return get_similarity(ENTITY_PREFIX + entity1, ENTITY_PREFIX + entity2)


def get_similarity(entry1, entry2):
    if entry1 == entry2:
        return 1.0
    return np.inner(get_norm_vector(entry1), get_norm_vector(entry2))


def cluster_match(cluster_entities, query_entities, min_similarity):
    for cluster_entity in cluster_entities:
        has_related_entity = False
        for query_entity in query_entities:
            if get_entity_similarity(query_entity, cluster_entity) > min_similarity:
                has_related_entity = True
                break
        if not has_related_entity:
            return False
    return True
