import numpy as np

EPSILON = 1e-3


def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm < EPSILON:
        return vector
    return vector / norm


def parse_entity_cluster(group):
    if group == "set()":
        group = list()
    else:
        group = group[2:-2].split("', '")
    group_entities = set()
    for entity in group:
        group_entities.add(entity)
    return group_entities
