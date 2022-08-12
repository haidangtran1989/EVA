from utils.config import *
from entity.wikivec_loader import get_entity_vector
import numpy as np


def extract_entities(text):
    begin = 0
    entities = set()
    while 0 <= begin < len(text):
        begin = text.find(WIKI_PATTERN, begin)
        if begin < 0:
            break
        entity_end = text.find(ENTITY_END_PATTERN, begin)
        entity = text[begin + len(WIKI_PATTERN):entity_end]
        begin = text.find(TAG_END_PATTERN, entity_end)
        entities.add(entity)
    return entities


def get_average_vector(entities):
    size = len(entities)
    sum_vector = np.zeros(ENTITY_DIMENSION)
    if size == 0:
        return sum_vector
    for entity in entities:
        sum_vector = np.add(sum_vector, get_entity_vector(entity))
    return sum_vector / size
